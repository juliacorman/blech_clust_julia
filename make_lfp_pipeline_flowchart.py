#!/usr/bin/env python3
"""Render the LFP processing and validation pipeline flowchart."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.textpath import TextPath

OUTPUT_BASE = "LFP_validation_pipeline_flowchart"

COLORS = {
    "blech": ("#E8F2F1", "#167C72"),
    "validate": ("#EAF0F8", "#3E6B9A"),
    "artifact": ("#FFF1E5", "#C56A2D"),
    "plot": ("#EEF4E8", "#668C3C"),
    "note": ("#FFF8DD", "#B38B18"),
}


def text_width_inches(text, font_properties):
    return TextPath((0, 0), text, prop=font_properties).get_extents().width / 72


def wrap_to_width(text, max_width, font_properties):
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if text_width_inches(candidate, font_properties) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        if text_width_inches(word, font_properties) <= max_width:
            current = word
            continue
        chunk = ""
        for character in word:
            candidate = chunk + character
            if text_width_inches(candidate, font_properties) <= max_width:
                chunk = candidate
            else:
                if chunk:
                    lines.append(chunk)
                chunk = character
        current = chunk
    if current:
        lines.append(current)
    return "\n".join(lines)

steps = [
    ("blech", "1. Raw recording and metadata",
     "Inputs are electrode .dat files, session .info/.params, dig-in events, trial timing, and electrode layout."),
    ("blech", "2. Upstream blech_clust processing",
     "Run initialization and trial bookkeeping, common-average reference, then blech processing. Post-processing makes arrays and runs QA."),
    ("blech", "3. Load or create Parsed_LFP",
     "get_lfps() reads /Parsed_LFP from HDF5. If absent, extract_lfps downsamples and bandpass-filters electrodes, epochs around dig-ins, and stores channels x trials x samples."),
    ("validate", "4. Map channels to regions",
     "Map each parsed LFP position to an electrode and implanted region. A channel must match exactly one region to be eligible."),
    ("validate", "5. Check each channel's signal",
     "Check finite, nonconstant samples. Flag flat-sample fraction > 1%, or robust-outlier fraction > 1% using |x - median| / (1.4826 x MAD) > 10."),
    ("validate", "6. Blech channel QC and peer comparison",
     "Run blech_clust trial QC per channel at MAD threshold 3. Compare median voltage, SD, and bad-trial fraction with same-region peers; deviations > 3 scaled MADs fail."),
    ("validate", "7. Approve analysis channels",
     "use_for_analysis is true only if signal checks pass, the region is valid, and regional artifact score passes. Store approved channel positions by region."),
    ("artifact", "8. Reject artifact trials",
     "remove_lfp_artifacts.py reruns validation with power QC off, then applies blech_clust to approved channels per taste. Keep a trial only if every channel passes: mean timewise deviation < median + 3 MADs."),
    ("artifact", "9. Save cleaned copies and audit",
     "Write retained trials to cleaned_LFP_artifacts/taste_NN_cleaned.npy. Record trial counts, approved positions, and manual taste exclusions. Original data stays unchanged. If none pass, warn and keep all trials."),
    ("plot", "10. Select tastes/channels before STFT",
     "Load cleaned arrays only for non-excluded tastes. Skip manually excluded tastes before STFT; select approved channels for region-specific plots."),
    ("plot", "11. STFT, baseline normalization, plots",
     "STFT: Fs 1000 Hz, 500-sample Hann window, overlap 499, max frequency 40 Hz. Z-score power per trial/channel/frequency against baseline, then average. Plot -1000 to +1750 ms with turbo."),
    ("plot", "12. Save session/grand-average figures",
     "Save region/taste spectrograms and pre/post frequency plots. grand_average_LFP.py combines baseline-normalized session results without changing original recordings."),
]

fig = plt.figure(figsize=(8.5, 11))
ax = fig.add_axes([0, 0, 1, 1])
fig.patch.set_facecolor("#F7F8FA")
ax.set_facecolor("#F7F8FA")
ax.set_xlim(0, 8.5)
ax.set_ylim(0, 11)
ax.axis("off")

fig.text(0.5, 0.975, "LFP Processing and Validation Pipeline",
         ha="center", va="top", fontsize=17, fontweight="bold", color="#182B3A")
fig.text(0.5, 0.948,
         "Blech_clust preprocessing precedes channel validation and trial cleaning",
         ha="center", va="top", fontsize=9.5, color="#4B5C68")

columns = [
    {"x": 0.35, "steps": steps[:6]},
    {"x": 4.6, "steps": steps[6:]},
]
box_width = 3.55
gap = 0.05
y_top = 9.52
centers = []

legend_items = [
    ("blech", "Blech_clust preprocessing"),
    ("validate", "Channel validation"),
    ("artifact", "Trial artifact removal"),
    ("plot", "Plotting"),
    ("note", "Review-only QC"),
]
legend_font = FontProperties(family="DejaVu Sans", size=6.8)
legend_swatch_width = 0.16
legend_gap = 0.11
legend_item_gap = 0.16
legend_width = sum(
    legend_swatch_width + legend_gap + text_width_inches(label, legend_font)
    for _, label in legend_items) + legend_item_gap * (len(legend_items) - 1)
legend_x = (8.5 - legend_width) / 2
for stage, label in legend_items:
    face, edge = COLORS[stage]
    ax.add_patch(FancyBboxPatch(
        (legend_x, 9.83), legend_swatch_width, 0.15,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        linewidth=1.0, edgecolor=edge, facecolor=face))
    label_x = legend_x + legend_swatch_width + legend_gap
    ax.text(label_x, 9.905, label, ha="left", va="center",
            fontsize=6.8, color="#263947")
    legend_x = label_x + text_width_inches(label, legend_font) + legend_item_gap

for column_index, column in enumerate(columns):
    x = column["x"]
    col_centers = []
    cursor = y_top
    for stage, title, body in column["steps"]:
        text_width = box_width - 0.3
        body_font = FontProperties(family="DejaVu Sans", size=8.2)
        title_font = FontProperties(
            family="DejaVu Sans", size=9.7, weight="bold")
        wrapped_title = wrap_to_width(title, text_width, title_font)
        wrapped_body = wrap_to_width(body, text_width, body_font)
        title_line_count = len(wrapped_title.splitlines())
        body_line_count = len(wrapped_body.splitlines())
        box_height = max(
            1.2,
            0.5 + title_line_count * 0.17 + body_line_count * 0.14)
        y = cursor - box_height
        face, edge = COLORS[stage]
        box = FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.4, edgecolor=edge, facecolor=face)
        ax.add_patch(box)
        ax.text(x + 0.14, y + box_height - 0.12, wrapped_title,
            ha="left", va="top", fontproperties=title_font,
            color="#193042", linespacing=1.05)
        body_top = y + box_height - 0.12 - title_line_count * 0.18 - 0.13
        ax.text(x + 0.14, body_top, wrapped_body,
            ha="left", va="top", fontproperties=body_font,
            color="#263947", linespacing=1.12)
        col_centers.append((x + box_width / 2, y, y + box_height))
        cursor = y - gap
    centers.append(col_centers)

    for current, following in zip(col_centers, col_centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (current[0], current[1] - 0.015),
            (following[0], following[2] + 0.015),
            arrowstyle="-|>", mutation_scale=9, linewidth=1.2,
            color="#536875"))

# Route the sequence from the bottom of the left column to the top of the right.
left_last = centers[0][-1]
right_first = centers[1][0]
route_y = 0.82
route_x = 8.28
right_first_mid = (right_first[1] + right_first[2]) / 2
ax.plot([left_last[0], left_last[0], route_x, route_x],
        [left_last[1], route_y, route_y, right_first_mid],
        color="#536875", linewidth=1.2)
ax.add_patch(FancyArrowPatch(
    (route_x, right_first_mid),
    (columns[1]["x"] + box_width, right_first_mid),
    arrowstyle="-|>", mutation_scale=9, linewidth=1.2, color="#536875"))

note_box = FancyBboxPatch(
    (0.4, 0.12), 7.7, 0.58,
    boxstyle="round,pad=0.025,rounding_size=0.06",
    linewidth=1.0, edgecolor=COLORS["note"][1], facecolor=COLORS["note"][0])
ax.add_patch(note_box)
ax.text(0.53, 0.61, "Review-only taste QC (not automatic exclusion)",
        ha="left", va="top", fontsize=8.8, fontweight="bold", color="#574716")
ax.text(0.53, 0.39,
        "RMS ratio >= 2 or optional STFT power deviation >= 2 prompts review. Manual taste exclusions are applied before plotting.\n"
        "Validation QC baseline: -1.00 to -0.25 s; plot baseline: -2.00 to -0.25 s. The 0.25 s buffer limits onset smearing.",
        ha="left", va="top", fontsize=7.1, color="#4C432C", linespacing=1.12)

fig.savefig(f"{OUTPUT_BASE}.jpg", dpi=300, facecolor=fig.get_facecolor())
fig.savefig(f"{OUTPUT_BASE}.svg", facecolor=fig.get_facecolor())
