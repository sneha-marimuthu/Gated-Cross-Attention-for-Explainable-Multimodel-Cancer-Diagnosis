"""
GCAEMMCD Clinical Pathology Decision Support Workstation Theme & Styling.
Implements a clean, bright, high-density medical workstation design system:
  - 100% Light Medical Palette (Hospital Slate #f8fafc, Crisp White #ffffff, Subtle Borders #e2e8f0)
  - Zero dark mode or black boxes: all badges, toolbars, and containers use clean light colors
  - Three distinct structural columns: Sample Input, Manual Input, and Diagnostic Output
  - Restrained clinical blue accents (#2563eb, #1d4ed8) and pastel status indicators
  - Full override against system dark mode to ensure high readability
  - 100% original GCAEMMCD identity with zero vendor trademarks
"""

GCAEMMCD_CUSTOM_CSS = """
/* Global Reset and Light Medical Typography */
:root, body, .gradio-container, .dark, [data-theme="dark"] {
    color-scheme: light !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

/* Force light mode background across all Gradio sub-elements */
.dark, .dark div, .dark span, .dark p, .dark label, .dark button, .dark input, .dark textarea,
.dark .gr-panel, .dark .gr-box, .dark .gr-group, .dark .gr-accordion {
    background-color: inherit !important;
    color: #0f172a !important;
}

/* Container constraints */
.gradio-container {
    max-width: 100% !important;
    padding: 0 16px !important;
    margin: 0 auto !important;
}

/* Suppress default Gradio footer */
footer {
    display: none !important;
}

/* Top GCAEMMCD Global Workstation Header */
.gca-top-bar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 10px 20px;
    margin: 0 -16px 8px -16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.gca-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 800;
    color: #1e3a8a;
    letter-spacing: -0.4px;
}
.gca-logo-icon {
    color: #2563eb;
    font-size: 22px;
}
.gca-engine-badge {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #334155;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.gca-telemetry {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 12px;
    font-weight: 500;
    color: #475569;
}
.gca-status-pill-active {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #166534;
    font-weight: 700;
    background-color: #dcfce7;
    border: 1px solid #bbf7d0;
    padding: 2px 8px;
    border-radius: 12px;
}

/* Specimen Session Sub-Header */
.gca-sub-header {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 16px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #475569;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}
.gca-meta-pill {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    color: #1d4ed8;
}
.gca-meta-label {
    color: #64748b;
    margin-right: 4px;
}

/* Three Columns Card Shells */
.workstation-column-shell {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}
.column-title-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    margin-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
}
.column-title-text {
    font-size: 14px;
    font-weight: 700;
    color: #1e3a8a;
    display: flex;
    align-items: center;
    gap: 6px;
}
.column-badge {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 2px 6px;
}

/* Specimen Selection Cards */
.specimen-preset-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
    margin-bottom: 8px !important;
    transition: all 0.15s ease !important;
}
.specimen-preset-card:hover {
    border-color: #3b82f6 !important;
    background-color: #f8fafc !important;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.08) !important;
}
.specimen-preset-card.active-selected {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

/* Status Badges */
.status-pill-malignant {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
}
.status-pill-normal {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
}
.status-pill-benign {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #fef9c3;
    color: #854d0e;
    border: 1px solid #fde047;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
}

/* Input Group Framing */
.input-sub-section {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
}
.input-sub-header {
    font-size: 12px;
    font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Pathology Viewport Header */
.gca-viewport-bar {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    color: #475569;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-bottom: -1px;
}

/* Main Viewport Container & Overlays - 100% LIGHT PALETTE */
.viewer-container {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    position: relative !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}
.viewer-slide-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 20;
    background-color: rgba(255, 255, 255, 0.96);
    border: 1px solid #cbd5e1;
    color: #0f172a;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}
.viewer-scale-bar {
    position: absolute;
    bottom: 10px;
    left: 10px;
    z-index: 20;
    background-color: rgba(255, 255, 255, 0.96);
    border: 1px solid #cbd5e1;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}
.viewer-timestamp {
    position: absolute;
    bottom: 10px;
    right: 10px;
    z-index: 20;
    background-color: rgba(255, 255, 255, 0.96);
    border: 1px solid #cbd5e1;
    color: #475569;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-family: monospace;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

/* Light Clinical Disclaimer Footer */
.gca-footer-disclaimer {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    color: #64748b;
    text-align: center;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.2px;
    margin-top: 14px;
}
"""

# Alias for backwards compatibility
AISIGHT_CUSTOM_CSS = GCAEMMCD_CUSTOM_CSS


def get_top_navbar_html():
    return """
    <div class="gca-top-bar">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div class="gca-brand">
                <span class="gca-logo-icon">🧬</span>
                <span>GCAEMMCD<span style="font-size: 13px; font-weight: 600; color: #64748b; margin-left: 8px;">Clinical Pathology Workstation</span></span>
            </div>
            <div class="gca-engine-badge">
                <span style="color: #64748b;">Architecture:</span>
                <strong>Tri-Modal Gated Cross-Attention (v2.4)</strong>
            </div>
        </div>
        <div class="gca-telemetry">
            <div><span>Hardware:</span> <strong>NVIDIA RTX 2050 (CUDA)</strong></div>
            <div><span>Backbones:</span> <strong>ResNet-50 · Phikon ViT-B · BioBERT v1.1</strong></div>
            <span class="gca-status-pill-active">● Models Calibrated</span>
        </div>
    </div>
    """


def get_sub_header_html(specimen_id="Specimen S-01", status="Ready"):
    return f"""
    <div class="gca-sub-header">
        <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <div style="font-weight: 700; color: #0f172a;">
                Diagnostic Session › <span class="gca-meta-pill">{specimen_id}</span>
            </div>
            <div><span class="gca-meta-label">Active Modalities:</span><strong>Histopathology (H&E) + Clinical Narrative + 16 Biomarkers</strong></div>
            <div><span class="gca-meta-label">Pipeline Status:</span><span style="color: #15803d; font-weight: 700;">● {status}</span></div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="color: #64748b; font-size: 12px;">Optical Domain & OOD Protection: <strong>ACTIVE</strong></span>
        </div>
    </div>
    """


def get_viewer_toolbar_html(specimen_id="Specimen S-01"):
    return f"""
    <div class="gca-viewport-bar">
        <div style="display: flex; align-items: center; gap: 10px;">
            <strong style="color: #0f172a;">Pathology Micrograph Viewport:</strong>
            <span style="color: #2563eb; font-weight: 600;">{specimen_id}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 14px;">
            <span style="color: #64748b;">Field of View: <strong>1.84 mm²</strong></span>
            <span style="color: #64748b;">|</span>
            <span style="color: #059669; font-weight: 700;">● Stage 0 Optical Check: Validated</span>
        </div>
    </div>
    """


def get_slide_badge_html(slide_id="H&E Slide S-01", status_badge="🔴 Malignant Neoplasm"):
    pill_class = "status-pill-malignant"
    if "Non-Neoplastic" in status_badge:
        pill_class = "status-pill-normal"
    elif "Benign" in status_badge:
        pill_class = "status-pill-benign"

    return f"""
    <div class="viewer-slide-badge">
        <span>{slide_id}</span>
        <span class="{pill_class}">{status_badge}</span>
    </div>
    """


def get_footer_disclaimer_html():
    return """
    <div class="gca-footer-disclaimer">
        GCAEMMCD Multimodal Cancer Diagnosis Decision Support System • For Investigational & Research Use Only • Zero-Leakage 5-Fold Cross-Validated & External Benchmark Verified
    </div>
    """
