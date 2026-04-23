import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec


def plot_basic_info(
    df: pd.DataFrame,
    *,
    fname: str | Path = "trial_dashboard.png",
    dpi: int = 150,
) -> Path:
    """
    Dashboard (Seaborn) – per‑trial dots + single Gantt panel.

        ┌──────────────┬──────────────┬──────────────┐
        │ Enrollment   │   Site #     │    Phase     │
        ├──────────────────── spans all cols ────────┤
        │   Gantt: Start → Primary Completion        │
        └─────────────────────────────────────────────┘
    Uses only the columns already in your scrape:
        • 'Enrollment Count'
        • 'Locations'  → on‑the‑fly site count
        • 'Phases'
        • 'Start Date', 'Primary Completion Date'
    """

    # ---------- Prep ---------------------------------------------------------
    df_plot = df.copy()

    # Parse dates
    date_cols = ["Start Date", "Primary Completion Date"]
    df_plot[date_cols] = (
    df_plot[date_cols]
    .apply(pd.to_datetime, errors="coerce", utc=True)   # utc=True optional
)

    fallback = ["#" + str(i) for i in df_plot.index]
    df_plot["Trial"] = df_plot["NCT ID"].fillna(pd.NA).combine_first(pd.Series(fallback))

    df_plot["Enrollment Count"] = pd.to_numeric(df_plot["Enrollment Count"], errors="coerce")
    df_plot["Site Count"] = (df_plot["Locations"].fillna("") .apply(lambda s: 0 if s.strip() == "" else s.count(";") + 1)
    )

    months = (df_plot["Primary Completion Date"] - df_plot["Start Date"]).dt.days / 30.44
    df_plot["EPSM"] = (df_plot["Enrollment Count"] /
                  (df_plot["Site Count"].replace({0: np.nan}) * months.replace({0: np.nan})))

    df_plot["Phase Label"] = df_plot["Phases"].str.split(",", n=1).str[0].str.strip()
    df_plot = df_plot.sort_values("Start Date", ascending=False).reset_index(drop=True)

    # ── Layout: 2 rows × 3 cols, bottom row spans all cols ---------------------
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 8), dpi=dpi)
    gs = GridSpec(
        nrows=2, ncols=3, figure=fig,
        height_ratios=[1, 2],  
        hspace=0.35, wspace=0.25,
    )

    ax_enroll = fig.add_subplot(gs[0, 0])
    ax_sites  = fig.add_subplot(gs[0, 1])
    ax_epsm  = fig.add_subplot(gs[0, 2])
    ax_gantt  = fig.add_subplot(gs[1, :])      

    # ── Panel 1: Enrollment histogram ---------------------------------------
    tail = 0.01                   # fraction to clip (1 %)
    cap  = df_plot["Enrollment Count"].quantile(1 - tail)
    sns.histplot(
        ax=ax_enroll,
        data=df_plot,
        # x=df_plot["Enrollment Count"].clip(upper=cap),
        x=df_plot["Enrollment Count"],
        discrete=True,
        bins="auto",            # let Seaborn/Mpl choose sensible breaks
        color=sns.color_palette("viridis")[3],
        edgecolor="white",
    )
    ax_enroll.set_title("Enrollment Distribution")
    ax_enroll.set_xlabel(f"Participants")
    ax_enroll.set_ylabel("Number of Trials")
    plt.setp(ax_enroll.get_xticklabels(), rotation=45, ha="right")

    # ── Panel 2: Site‑count histogram ---------------------------------------
    tail = 0.10                    # fraction to clip (1 %)
    cap  = df_plot["Site Count"].quantile(1 - tail)
    sns.histplot(
        ax=ax_sites,
        data=df_plot,
        # x=df_plot["Site Count"].clip(upper=cap),
        x=df_plot["Site Count"],
        discrete=True,          # 1,2,3… exact bars
        # binwidth=1,
        # bins="auto",
        color=sns.color_palette("crest")[3],
        edgecolor="white",
    )
    ax_sites.set_title("Site‑Count Distribution")
    ax_sites.set_xlabel(f"Sites")
    ax_sites.set_ylabel("Number of Trials")
    plt.setp(ax_sites.get_xticklabels(), rotation=45, ha="right")


    # ---------- Phase panel --------------------------------------------------
    sns.histplot(
        ax=ax_epsm,
        data=df_plot,
        x="EPSM",
        binwidth=1,
        bins="auto",
        color=sns.color_palette("crest")[3],
        edgecolor="white",
    )

    ax_epsm.set_title("EPSM Distribution")
    ax_epsm.set_xlabel("Enrollment per Site per Month (EPSM)")
    ax_epsm.set_ylabel("Number of Trials")

    # ---------- Gantt panel --------------------------------------------------
    ax_gantt.set_title("Enrollment Timeline")
    for _, row in df_plot.iterrows():
        if pd.notna(row["Start Date"]) and pd.notna(row["Primary Completion Date"]):
            ax_gantt.barh(
                y=row["Trial"],
                left=row["Start Date"],
                width=row["Primary Completion Date"] - row["Start Date"],
                height=0.45,
                color=sns.color_palette("deep")[0],
                alpha=0.8,
            )
    ax_gantt.set_xlabel("Date")
    ax_gantt.set_ylabel("")
    ax_gantt.invert_yaxis()
    ax_gantt.xaxis_date()

    # ---------- Finish up ----------------------------------------------------
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    plt.close(fig)
    return Path(fname)

def plot_criterion_info(df: pd.DataFrame,
                        fname: str | Path = "trial_dashboard.png",
                        dpi: int = 150) -> Path:
    """
    Dashboard – bar‑value panels + Gantt timeline
      ┌──────────────┬──────────────┬──────────────────────────┐
      │ Enrollment   │   Site #     │ Enrol / Site / Month ★   │
      ├──────────────┴──────────────┴──────────────────────────┤
      │ Gantt: Start → Primary Completion                      │
      └─────────────────────────────────────────────────────────┘
    """

    df_plot = df.copy()
    date_cols = ["Start Date", "Primary Completion Date"]
    df_plot[date_cols] = df_plot[date_cols].apply(
        pd.to_datetime, errors="coerce", utc=True
    )

    # Trial label
    fallback = ["#" + str(i) for i in df_plot.index]
    df_plot["Trial"] = (
        df_plot.get("NCT ID", pd.Series(dtype=object))
               .fillna(pd.NA)
               .combine_first(pd.Series(fallback))
    )

    # Compute metrics
    df_plot["Enrollment Count"] = pd.to_numeric(
        df_plot["Enrollment Count"], errors="coerce"
    )
    df_plot["Site Count"] = (
        df_plot["Locations"].fillna("")
              .apply(lambda s: 0 if s.strip() == "" else s.count(";") + 1)
    )
    months = (
        (df_plot["Primary Completion Date"] - df_plot["Start Date"])
        .dt.days / 30.44
    )
    df_plot["EPSM"] = (
        df_plot["Enrollment Count"] /
        (df_plot["Site Count"].replace({0: np.nan}) * months.replace({0: np.nan}))
    )

    # Plot
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 8), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2], hspace=0.5, wspace=0.25)

    ax_enroll = fig.add_subplot(gs[0, 0])
    ax_sites  = fig.add_subplot(gs[0, 1])
    ax_epsm   = fig.add_subplot(gs[0, 2])
    ax_gantt  = fig.add_subplot(gs[1, :])

    # Barplot for each trial (not histogram)
    def value_barplot(ax, x_vals, y_vals, title, ylabel, color):
        sns.barplot(x=x_vals, y=y_vals, ax=ax, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Trial")
        ax.tick_params(axis="x", rotation=45)

    value_barplot(ax_enroll, df_plot["Trial"], df_plot["Enrollment Count"],
                  "Enrollment Count by Trial", "Participants (n)", sns.color_palette("viridis")[3])
    value_barplot(ax_sites, df_plot["Trial"], df_plot["Site Count"],
                  "Site Count by Trial", "Sites", sns.color_palette("crest")[3])
    value_barplot(ax_epsm, df_plot["Trial"], df_plot["EPSM"],
                  "EPSM by Trial", "Enrollment / Site / Month", sns.color_palette("mako")[2])

    # Gantt plot
    ax_gantt.set_title("Enrollment Timeline")
    for _, row in df_plot.iterrows():
        if pd.notna(row["Start Date"]) and pd.notna(row["Primary Completion Date"]):
            ax_gantt.barh(
                y=row["Trial"],
                left=row["Start Date"],
                width=row["Primary Completion Date"] - row["Start Date"],
                height=0.45,
                color=sns.color_palette("deep")[0],
                alpha=0.8,
            )
    ax_gantt.set_xlabel("Date")
    ax_gantt.set_ylabel("")
    ax_gantt.invert_yaxis()
    ax_gantt.xaxis_date()
    ax_gantt.grid(True, axis="x", ls="--", alpha=.25)

    # Save
    plt.tight_layout()
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)
    return Path(fname)