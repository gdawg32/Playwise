# utils/xi_pitch.py
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from io import BytesIO
import matplotlib.patches as patches

# Luxury pitch with premium aesthetics
pitch = Pitch(
    pitch_type='opta',
    pitch_color='#fafbfc',      # Softer white
    line_color='#c5cdd4',       # Refined gray
    linewidth=2.0,              # Slightly bolder lines
    line_zorder=1,
    stripe=True,                # Subtle stripes
    stripe_color='#f5f7f9'      # Very subtle stripe
)

formation_df = pitch.formations_dataframe

SLOT_PRIORITY = {
    'GK':  ['GK'],
    'LB':  ['LB', 'LWB'],
    'RB':  ['RB', 'RWB'],
    'CB':  ['LCB', 'RCB', 'CB', 'SW'],
    'LWB': ['LWB', 'LB'],
    'RWB': ['RWB', 'RB'],
    'CDM': ['LDM', 'RDM', 'CDM'],
    'CM':  ['LCM', 'RCM', 'CM', 'LDM', 'RDM', 'CAM'],
    'LM':  ['LM', 'LW', 'LAM'],
    'RM':  ['RM', 'RW', 'RAM'],
    'CAM': ['LAM', 'RAM', 'CAM'],
    'LW':  ['LW', 'LM', 'LAM', 'LF'],
    'RW':  ['RW', 'RM', 'RAM', 'RF'],
    'ST':  ['ST', 'CF', 'LCF', 'RCF', 'SS'],
}

def normalize_formation(name):
    return name.replace("-", "").lower()

def get_formation_coords(formation):
    key = normalize_formation(formation)
    subset = formation_df[formation_df["formation"] == key]
    if subset.empty:
        return None
    return {r["name"]: (r["x"], r["y"]) for _, r in subset.iterrows()}

def draw_xi(team_name, formation, lineup, title_suffix):
    coords = get_formation_coords(formation)
    if not coords:
        raise ValueError("Formation not supported")

    remaining = coords.copy()
    fig, ax = pitch.draw(figsize=(12, 8))
    
    # Add subtle shadow/depth to the pitch
    ax.set_facecolor('#fafbfc')
    
    # Premium border frame
    frame = patches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor='#e1e4e8',
        linewidth=3,
        zorder=0
    )
    ax.add_patch(frame)

    for p in lineup:
        slot = p["slot"]
        name = p["player_name"]

        assigned = None
        if slot in remaining:
            assigned = slot
        else:
            for alt in SLOT_PRIORITY.get(slot, []):
                if alt in remaining:
                    assigned = alt
                    break

        if not assigned:
            continue

        x, y = remaining.pop(assigned)

        # Luxury player dot with gradient effect and premium shadow
        pitch.scatter(
            x, y, ax=ax,
            s=520,                          # Slightly larger
            color='#6b0f1a',                # Deep burgundy
            edgecolors='#1a1a1a',           # Rich black edge
            linewidth=2.5,                  # Premium border
            alpha=0.95,                     # Slight transparency
            zorder=3
        )
        
        # Inner highlight for depth
        pitch.scatter(
            x, y, ax=ax,
            s=240,
            color='#8b1820',                # Lighter burgundy center
            edgecolors='none',
            alpha=0.6,
            zorder=4
        )

        # Premium name label with refined styling
        pitch.annotate(
            name.replace(" ", "\n"),
            (x, y - 5),
            ax=ax,
            ha="center",
            va="top",
            fontsize=10.5,
            weight="600",                    # Semi-bold
            fontfamily='sans-serif',
            color='#1a1a1a',                # Deep black text
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc='#ffffff',                # Pure white background
                ec='#d1d5db',                # Soft gray border
                linewidth=1.5,
                alpha=0.98
            ),
            zorder=5
        )

    # Premium title with refined typography
    title_text = f"{team_name}  •  {formation}  •  {title_suffix}"
    ax.text(
        0.5, 1.06,
        title_text,
        transform=ax.transAxes,
        fontsize=17,
        weight='700',
        ha='center',
        va='top',
        fontfamily='sans-serif',
        color='#1a1a1a',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='#f8f9fa',
            edgecolor='#e1e4e8',
            linewidth=2,
            alpha=0.95
        )
    )
    
    # Subtle watermark/branding corner
    ax.text(
        0.98, 0.02,
        'Tactical Analysis',
        transform=ax.transAxes,
        fontsize=8,
        weight='500',
        ha='right',
        va='bottom',
        color='#9ca3af',
        alpha=0.6,
        style='italic'
    )

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=180,                    # Higher DPI for sharper output
        bbox_inches="tight",
        facecolor='#fafbfc',        # Match pitch background
        edgecolor='none',
        pad_inches=0.3              # Clean padding
    )
    plt.close(fig)
    buf.seek(0)
    return buf
