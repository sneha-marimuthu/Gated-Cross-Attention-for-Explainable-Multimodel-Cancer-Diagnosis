"""
PathAI AISight Clinical Workstation Theme & Styling for GCAEMMCD.
Faithfully replicates the visual identity of PathAI AISight:
  - Clean hospital clinical white/slate palette (#f8fafc / #ffffff / #0f172a)
  - Top global clinical header with red hexagonal AISight® branding, site selector, and pathologist profile
  - Accession patient metadata breadcrumb bar (Created, DOB, Status, Preview Report, etc.)
  - Digital pathology toolbar (Undo, Redo, Zoom, Caliper, Shapes, Pins, Split, Minimap)
  - Left slide carousel / thumbnail tray with status pill tags ('Possible cancer', 'Non-Neoplastic', 'Benign')
  - Central WSI viewer container with floating top-left slide badge, floating scale bar, and minimap
  - Collapsible clinical inspector tree ('Slide Acceptability', 'Algorithm Results', 'Alignments', 'XAI Audit')
  - Translucent research disclaimer footer
"""

AISIGHT_CUSTOM_CSS = """
/* Global Reset and Clinical Typography */
body, .gradio-container {
    background-color: #f1f5f9 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    color: #0f172a !important;
}

/* Suppress default Gradio footer and header padding */
footer {
    display: none !important;
}
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Top AISight Global Navigation Bar */
.aisight-top-bar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 10px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.aisight-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.4px;
}
.aisight-logo-icon {
    color: #e11d48;
    font-size: 24px;
    font-weight: 900;
}
.aisight-site-badge {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 13px;
    color: #334155;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.aisight-nav-links {
    display: flex;
    align-items: center;
    gap: 22px;
    font-size: 13px;
    font-weight: 500;
    color: #475569;
}
.aisight-nav-link {
    color: #475569;
    text-decoration: none;
    cursor: pointer;
    transition: color 0.15s;
}
.aisight-nav-link:hover, .aisight-nav-link.active {
    color: #0f172a;
    font-weight: 600;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 2px;
}
.aisight-user-area {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 13px;
    font-weight: 600;
    color: #1e293b;
}

/* Sub-Header / Accession Metadata Bar */
.aisight-sub-header {
    background-color: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    padding: 8px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #475569;
}
.aisight-meta-pill {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    color: #0f172a;
}
.aisight-meta-label {
    color: #64748b;
    margin-right: 4px;
}

/* Digital Pathology Toolbar */
.aisight-toolbar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 6px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    color: #475569;
}
.aisight-tool-btn {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 14px;
    cursor: pointer;
    color: #475569;
    transition: all 0.15s ease;
}
.aisight-tool-btn:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
    color: #0f172a;
}
.aisight-tool-separator {
    width: 1px;
    height: 18px;
    background-color: #e2e8f0;
    margin: 0 4px;
}

/* Slide Tray on Left */
.slide-tray-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    margin-bottom: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.slide-tray-card:hover {
    border-color: #3b82f6 !important;
    box-shadow: 0 2px 6px rgba(59, 130, 246, 0.12) !important;
}
.slide-tray-card.selected {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}
.slide-pill-cancer {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
}
.slide-pill-normal {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
}
.slide-pill-benign {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background-color: #fef9c3;
    color: #854d0e;
    border: 1px solid #fef08a;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
}

/* Main Viewport Container */
.viewer-container {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    position: relative !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}
.viewer-slide-badge {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 20;
    background-color: rgba(15, 23, 42, 0.88);
    backdrop-filter: blur(6px);
    color: #ffffff;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}
.viewer-scale-bar {
    position: absolute;
    bottom: 12px;
    left: 12px;
    z-index: 20;
    background-color: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(6px);
    border: 1px solid #cbd5e1;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
}
.viewer-timestamp {
    position: absolute;
    bottom: 12px;
    right: 12px;
    z-index: 20;
    background-color: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(6px);
    color: #ffffff;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-family: monospace;
}

/* Translucent Footer Disclaimer */
.aisight-footer-disclaimer {
    background-color: rgba(15, 23, 42, 0.92);
    backdrop-filter: blur(6px);
    color: #94a3b8;
    text-align: center;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.2px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 12px;
}
"""

def get_top_navbar_html():
    return """
    <div class="aisight-top-bar">
        <div style="display: flex; align-items: center; gap: 18px;">
            <div class="aisight-brand">
                <span class="aisight-logo-icon">⬡</span>
                <span>AISight<sup style="font-size: 10px; font-weight: 500; color: #64748b;">®</sup></span>
            </div>
            <div class="aisight-site-badge">
                <span style="color: #64748b;">Site:</span>
                <strong>PathAI Demo Site, Boston, MA</strong>
                <span style="font-size: 10px; color: #94a3b8;">▼</span>
            </div>
        </div>
        <div class="aisight-nav-links">
            <span class="aisight-nav-link active">Case Assignments</span>
            <span class="aisight-nav-link">Accessions</span>
            <span class="aisight-nav-link">Library</span>
            <span class="aisight-nav-link">Audit Trail</span>
            <span class="aisight-nav-link">Users</span>
            <span class="aisight-nav-link">Help</span>
        </div>
        <div class="aisight-user-area">
            <span style="font-size: 15px;">🇺🇸</span>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="background-color: #2563eb; color: #fff; width: 26px; height: 26px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">RB</span>
                <span>Rishi Bhargava ▾</span>
            </div>
            <span style="color: #64748b; font-size: 16px; cursor: pointer;">🔔</span>
            <span style="color: #64748b; font-size: 16px; cursor: pointer;">⠿</span>
        </div>
    </div>
    """

def get_sub_header_html(accession_id="ACC-2026-9041", status="Assessment Complete"):
    return f"""
    <div class="aisight-sub-header">
        <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <div style="font-weight: 700; color: #0f172a;">
                <span style="color: #2563eb; cursor: pointer;">Case Assignments</span> › Accession <span class="aisight-meta-pill">{accession_id}</span>
            </div>
            <div><span class="aisight-meta-label">Created:</span><strong>2026-11-13</strong></div>
            <div><span class="aisight-meta-label">Name:</span><strong>Confidential / Anonymized</strong></div>
            <div><span class="aisight-meta-label">DOB:</span><strong>1962-04-18 (64 yo)</strong></div>
            <div><span class="aisight-meta-label">Accession Status:</span><span style="color: #15803d; font-weight: 700;">● {status}</span></div>
            <div><span class="aisight-meta-label">Pending Actions:</span><strong style="color: #2563eb;">Pathologist Sign-Out ⚡</strong></div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="background-color: #2563eb; color: #ffffff; padding: 5px 12px; border-radius: 5px; font-weight: 600; font-size: 12px; cursor: pointer;">📄 Preview Report</span>
            <span style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 4px 8px; border-radius: 5px; cursor: pointer;">‹</span>
            <span style="background-color: #ffffff; border: 1px solid #cbd5e1; padding: 4px 8px; border-radius: 5px; cursor: pointer;">›</span>
            <span style="color: #64748b; font-weight: bold; cursor: pointer;">•••</span>
            <span style="color: #2563eb; font-weight: 600; font-size: 12px; margin-left: 8px; cursor: pointer;">💬 Accession Discussion</span>
        </div>
    </div>
    """

def get_viewer_toolbar_html():
    return """
    <div class="aisight-toolbar">
        <div style="display: flex; align-items: center; gap: 6px;">
            <button class="aisight-tool-btn" title="Undo">↶</button>
            <button class="aisight-tool-btn" title="Redo">↷</button>
            <div class="aisight-tool-separator"></div>
            <button class="aisight-tool-btn" title="Reset View">⛶</button>
            <button class="aisight-tool-btn" title="Zoom In">🔍+</button>
            <button class="aisight-tool-btn" title="Zoom Out">🔍-</button>
            <div class="aisight-tool-separator"></div>
            <button class="aisight-tool-btn" title="Snapshot Micrograph">📷</button>
            <button class="aisight-tool-btn" title="Caliper Ruler">📏</button>
            <button class="aisight-tool-btn" title="Polygon Annotation">⬠</button>
            <button class="aisight-tool-btn" title="Ellipse Annotation">◯</button>
            <button class="aisight-tool-btn" title="Rectangle Annotation">▢</button>
            <button class="aisight-tool-btn" title="Freehand Pen">✎</button>
            <div class="aisight-tool-separator"></div>
            <button class="aisight-tool-btn" title="Center Focus">🎯</button>
            <button class="aisight-tool-btn" title="Add Diagnostic Pin">📍</button>
            <button class="aisight-tool-btn" title="Dual Split View">◫</button>
            <button class="aisight-tool-btn" title="Toggle Macro Minimap">🗺️</button>
        </div>
        <div style="display: flex; align-items: center; gap: 10px; font-size: 12px;">
            <span style="color: #64748b;">Field of View: <strong>1.84 mm²</strong></span>
            <span style="color: #64748b;">|</span>
            <span style="color: #059669; font-weight: 700;">● Stage 0 Optical Check: Valid</span>
        </div>
    </div>
    """

def get_slide_badge_html(slide_id="H&E Slide A1-1", status_badge="🔴 Possible cancer"):
    pill_class = "slide-pill-cancer"
    if "Non-Neoplastic" in status_badge:
        pill_class = "slide-pill-normal"
    elif "Benign" in status_badge:
        pill_class = "slide-pill-benign"

    return f"""
    <div class="viewer-slide-badge">
        <span>{slide_id}</span>
        <span class="{pill_class}">{status_badge}</span>
    </div>
    """

def get_footer_disclaimer_html():
    return """
    <div class="aisight-footer-disclaimer">
        AISight® is for Research Use Only. Not for use in diagnostic procedures. • Mass General Brigham & Dana-Farber Integrated Trial • GCAEMMCD Multimodal Engine
    </div>
    """
