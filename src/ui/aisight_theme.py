"""
GCAEMMCD Clinical Pathology Decision Support Workstation Theme & Styling.
Implements a punchy, ultra-clean light medical workstation design system:
  - 100% Light Medical Palette (Pure White #ffffff, Hospital Slate #f8fafc)
  - Punchy Accent System:
      * Royal & Cobalt Blue (#2563eb, #1d4ed8) for primary CTAs, active cards, and branding
      * Emerald & Mint Green (#10b981, #059669) for verified states, optical domain checks, and non-neoplastic markers
      * Punchy Orange & Coral (#f97316, #ea580c, #c2410c) for malignant neoplasm pills, alerts, and focal highlights
      * Warm Amber (#f59e0b) for benign lesion badges
  - Zero Dark Mode:
      * JavaScript MutationObserver permanently purges the 'dark' class
      * Universal CSS overrides force light backgrounds and dark readable text (#0f172a) everywhere
  - 3-Column Fixed-Height Architecture with Column Inner Scroll:
      * Dynamic layout: flex-based responsive width proportions
      * Fixed height: calc(100vh - 110px) with independent column vertical scrolling (no page jumping)
      * Custom punchy scrollbars
"""

import gradio as gr

# Force Light Mode & Workstation Column Resizers JavaScript
FORCE_LIGHT_MODE_JS = """
() => {
    function enforceLight() {
        try {
            document.documentElement.classList.remove('dark');
            if (document.body) document.body.classList.remove('dark');
            document.documentElement.setAttribute('data-theme', 'light');
            document.documentElement.style.colorScheme = 'light';
            localStorage.setItem('theme', 'light');
            localStorage.setItem('color-theme', 'light');
        } catch(e) {}
    }

    function initWorkstationResizers() {
        try {
            if (document.getElementById('col-resizer-1') && document.getElementById('col-resizer-2')) {
                return;
            }
            const container = document.querySelector('.workstation-row-container');
            if (!container) return;
            const col1 = container.querySelector('.col-sample-input');
            const col2 = container.querySelector('.col-manual-input');
            const col3 = container.querySelector('.col-diagnostic-output');
            if (!col1 || !col2 || !col3) return;

            // Restore previously saved widths if present
            try {
                const savedW1 = sessionStorage.getItem('gca_col1_width');
                if (savedW1) {
                    col1.style.setProperty('width', savedW1 + 'px', 'important');
                    col1.style.setProperty('flex', '0 0 ' + savedW1 + 'px', 'important');
                }
                const savedW2 = sessionStorage.getItem('gca_col2_width');
                if (savedW2) {
                    col2.style.setProperty('width', savedW2 + 'px', 'important');
                    col2.style.setProperty('flex', '0 0 ' + savedW2 + 'px', 'important');
                }
            } catch(e) {}

            // Resizer 1 (Between Column 1 and Column 2)
            if (!document.getElementById('col-resizer-1')) {
                const resizer1 = document.createElement('div');
                resizer1.id = 'col-resizer-1';
                resizer1.className = 'col-width-resizer col-resizer-1';
                resizer1.setAttribute('title', 'Drag to adjust column width · Double-click to reset');
                resizer1.innerHTML = '<div class="resizer-handle-grip"><span></span><span></span><span></span></div>';
                col1.after(resizer1);

                function onMouseDown1(e) {
                    e.preventDefault();
                    const startX = e.clientX || (e.touches && e.touches[0].clientX);
                    const startW = col1.getBoundingClientRect().width;
                    resizer1.classList.add('is-dragging');
                    document.body.style.cursor = 'col-resize';
                    document.body.style.userSelect = 'none';

                    function onMouseMove(ev) {
                        const clientX = ev.clientX || (ev.touches && ev.touches[0].clientX);
                        if (!clientX) return;
                        const delta = clientX - startX;
                        const containerW = container.getBoundingClientRect().width;
                        const maxW = Math.min(500, containerW * 0.45);
                        const newW = Math.max(170, Math.min(maxW, startW + delta));
                        col1.style.setProperty('width', newW + 'px', 'important');
                        col1.style.setProperty('flex', '0 0 ' + newW + 'px', 'important');
                    }

                    function onMouseUp() {
                        resizer1.classList.remove('is-dragging');
                        document.body.style.cursor = '';
                        document.body.style.userSelect = '';
                        document.removeEventListener('mousemove', onMouseMove);
                        document.removeEventListener('mouseup', onMouseUp);
                        document.removeEventListener('touchmove', onMouseMove);
                        document.removeEventListener('touchend', onMouseUp);
                        try {
                            sessionStorage.setItem('gca_col1_width', Math.round(col1.getBoundingClientRect().width));
                        } catch(e) {}
                    }

                    document.addEventListener('mousemove', onMouseMove);
                    document.addEventListener('mouseup', onMouseUp);
                    document.addEventListener('touchmove', onMouseMove, {passive: false});
                    document.addEventListener('touchend', onMouseUp);
                }

                resizer1.addEventListener('mousedown', onMouseDown1);
                resizer1.addEventListener('touchstart', onMouseDown1, {passive: false});
                resizer1.addEventListener('dblclick', () => {
                    col1.style.removeProperty('width');
                    col1.style.removeProperty('flex');
                    try { sessionStorage.removeItem('gca_col1_width'); } catch(e) {}
                });
            }

            // Resizer 2 (Between Column 2 and Column 3)
            if (!document.getElementById('col-resizer-2')) {
                const resizer2 = document.createElement('div');
                resizer2.id = 'col-resizer-2';
                resizer2.className = 'col-width-resizer col-resizer-2';
                resizer2.setAttribute('title', 'Drag to adjust column width · Double-click to reset');
                resizer2.innerHTML = '<div class="resizer-handle-grip"><span></span><span></span><span></span></div>';
                col2.after(resizer2);

                function onMouseDown2(e) {
                    e.preventDefault();
                    const startX = e.clientX || (e.touches && e.touches[0].clientX);
                    const startW = col2.getBoundingClientRect().width;
                    resizer2.classList.add('is-dragging');
                    document.body.style.cursor = 'col-resize';
                    document.body.style.userSelect = 'none';

                    function onMouseMove(ev) {
                        const clientX = ev.clientX || (ev.touches && ev.touches[0].clientX);
                        if (!clientX) return;
                        const delta = clientX - startX;
                        const containerW = container.getBoundingClientRect().width;
                        const col1W = col1.getBoundingClientRect().width;
                        const maxW = Math.min(850, containerW - col1W - 320);
                        const newW = Math.max(280, Math.min(maxW, startW + delta));
                        col2.style.setProperty('width', newW + 'px', 'important');
                        col2.style.setProperty('flex', '0 0 ' + newW + 'px', 'important');
                    }

                    function onMouseUp() {
                        resizer2.classList.remove('is-dragging');
                        document.body.style.cursor = '';
                        document.body.style.userSelect = '';
                        document.removeEventListener('mousemove', onMouseMove);
                        document.removeEventListener('mouseup', onMouseUp);
                        document.removeEventListener('touchmove', onMouseMove);
                        document.removeEventListener('touchend', onMouseUp);
                        try {
                            sessionStorage.setItem('gca_col2_width', Math.round(col2.getBoundingClientRect().width));
                        } catch(e) {}
                    }

                    document.addEventListener('mousemove', onMouseMove);
                    document.addEventListener('mouseup', onMouseUp);
                    document.addEventListener('touchmove', onMouseMove, {passive: false});
                    document.addEventListener('touchend', onMouseUp);
                }

                resizer2.addEventListener('mousedown', onMouseDown2);
                resizer2.addEventListener('touchstart', onMouseDown2, {passive: false});
                resizer2.addEventListener('dblclick', () => {
                    col2.style.removeProperty('width');
                    col2.style.removeProperty('flex');
                    try { sessionStorage.removeItem('gca_col2_width'); } catch(e) {}
                });
            }
        } catch(e) {}
    }

    function tryInit() {
        if (!document.getElementById('col-resizer-1') || !document.getElementById('col-resizer-2')) {
            initWorkstationResizers();
        }
    }

    enforceLight();
    tryInit();
    setTimeout(tryInit, 200);
    setTimeout(tryInit, 600);
    setTimeout(tryInit, 1500);
    window.addEventListener('DOMContentLoaded', tryInit);
    window.addEventListener('load', tryInit);

    const observer = new MutationObserver(() => {
        enforceLight();
        tryInit();
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
}
"""

GCAEMMCD_CUSTOM_CSS = """
/* -------------------------------------------------------------------------
   1. GLOBAL RESET & ZERO-DARK OVERRIDES
   ------------------------------------------------------------------------- */
:root, html, body, .gradio-container, .dark, [data-theme="dark"],
.dark *, [data-theme="dark"] * {
    color-scheme: light !important;
}

html, body {
    height: 100% !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

body, .gradio-container {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    max-width: 100% !important;
    height: 100vh !important;
    max-height: 100vh !important;
    padding: 0 12px 4px 12px !important;
    margin: 0 !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Force light backgrounds & dark text across ALL Gradio elements */
.dark, [data-theme="dark"],
.dark body, .dark .gradio-container,
.dark .gr-panel, .dark .gr-box, .dark .gr-form,
.dark .block, .dark .gr-group, .dark .gr-accordion,
.dark input, .dark textarea, .dark select,
.dark .gr-radio, .dark .gr-checkbox,
.dark .tabs, .dark .tabitem,
.dark .gr-button-secondary {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #e2e8f0 !important;
}

/* Typography & Prose: 100% Dark High-Contrast Slate Text */
.prose, .prose *, .prose p, .prose span, .prose li, .prose strong, .prose h1, .prose h2, .prose h3, .prose h4, .prose h5,
.markdown, .markdown *, .gr-accordion, .gr-accordion *,
.dark .prose, .dark .prose *, .dark .markdown, .dark .markdown *,
.dark .gr-accordion, .dark .gr-accordion *,
.dark p, .dark span, .dark li, .dark strong, .dark em, .dark h1, .dark h2, .dark h3, .dark h4, .dark h5, .dark h6 {
    color: #0f172a !important;
}

/* Highlighted code snippets */
code {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    padding: 1px 4px !important;
    border-radius: 4px !important;
}

/* Suppress default Gradio footer */
footer {
    display: none !important;
}

/* -------------------------------------------------------------------------
   2. TOP GLOBAL WORKSTATION HEADER & SUB-HEADER
   ------------------------------------------------------------------------- */
.gca-top-bar {
    background-color: #ffffff !important;
    border-bottom: 2.5px solid #2563eb !important;
    padding: 8px 18px !important;
    margin: 0 -16px 8px -16px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: space-between !important;
    align-items: center !important;
    gap: 10px !important;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.05) !important;
}
.gca-brand {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 10px !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    color: #1d4ed8 !important;
    letter-spacing: -0.3px !important;
}
.gca-logo-icon {
    color: #2563eb !important;
    font-size: 22px !important;
}
.gca-engine-badge {
    background-color: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 6px !important;
    padding: 3px 10px !important;
    font-size: 11px !important;
    color: #1e40af !important;
    font-weight: 700 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.gca-telemetry {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
.gca-status-pill-active {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    color: #15803d !important;
    font-weight: 800 !important;
    background-color: #dcfce7 !important;
    border: 1.5px solid #86efac !important;
    padding: 2px 10px !important;
    border-radius: 12px !important;
}

/* Specimen Session Sub-Header */
.gca-sub-header {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    margin-bottom: 10px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: space-between !important;
    align-items: center !important;
    gap: 8px !important;
    font-size: 12px !important;
    color: #334155 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}
.gca-meta-pill {
    background-color: #eff6ff !important;
    border: 1px solid #93c5fd !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-weight: 800 !important;
    color: #1d4ed8 !important;
}

/* -------------------------------------------------------------------------
   3. THREE COLUMNS WORKSPACE: COLUMN WIDTH ADJUSTERS & INDEPENDENT INTERNAL SCROLL
   ------------------------------------------------------------------------- */
.workstation-row-container {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 0 !important;
    flex: 1 1 auto !important;
    height: calc(100vh - 102px) !important;
    max-height: calc(100vh - 102px) !important;
    min-height: 480px !important;
    margin-top: 2px !important;
    margin-bottom: 2px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    width: 100% !important;
    position: relative !important;
}

/* Column 1: Sample Input (↕ Independent Internal Scroll) */
.col-sample-input {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    flex: 0 0 240px !important;
    width: 240px !important;
    min-width: 170px !important;
    max-width: 480px !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
    scrollbar-width: thin !important;
    scrollbar-color: #2563eb #eff6ff !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    box-sizing: border-box !important;
}

/* Column 2: Manual Input (↕ Independent Internal Scroll) */
.col-manual-input {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    flex: 0 0 470px !important;
    width: 470px !important;
    min-width: 280px !important;
    max-width: 850px !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
    scrollbar-width: thin !important;
    scrollbar-color: #2563eb #eff6ff !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    box-sizing: border-box !important;
}

/* Column 3: Output (↕ Independent Internal Scroll) */
.col-diagnostic-output {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    flex: 1 1 auto !important;
    min-width: 320px !important;
    height: 100% !important;
    max-height: 100% !important;
    min-height: 0 !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
    scrollbar-width: thin !important;
    scrollbar-color: #2563eb #eff6ff !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    box-sizing: border-box !important;
}

/* Column Width Adjuster Gutters (Interactive Splitters) */
.col-width-resizer {
    flex: 0 0 12px !important;
    width: 12px !important;
    min-width: 12px !important;
    max-width: 12px !important;
    cursor: col-resize !important;
    background: transparent !important;
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 50 !important;
    user-select: none !important;
    -webkit-user-select: none !important;
    margin: 0 2px !important;
    box-sizing: border-box !important;
}

/* Subtle Vertical Divider Line */
.col-width-resizer::before {
    content: '' !important;
    position: absolute !important;
    top: 6px !important;
    bottom: 6px !important;
    left: 5px !important;
    width: 2px !important;
    background-color: #cbd5e1 !important;
    border-radius: 1px !important;
    transition: background-color 0.15s ease, width 0.15s ease !important;
}

/* Centered Grip Handle with Tactile Indicator Dots */
.resizer-handle-grip {
    width: 8px !important;
    height: 38px !important;
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 4px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 3px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    z-index: 52 !important;
    pointer-events: none !important;
    transition: all 0.15s ease !important;
}
.resizer-handle-grip span {
    display: block !important;
    width: 2.5px !important;
    height: 2.5px !important;
    background-color: #64748b !important;
    border-radius: 50% !important;
}

/* Active Hover & Drag Highlight */
.col-width-resizer:hover::before,
.col-width-resizer.is-dragging::before {
    background-color: #2563eb !important;
    width: 3px !important;
}
.col-width-resizer:hover .resizer-handle-grip,
.col-width-resizer.is-dragging .resizer-handle-grip {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    transform: scale(1.1) !important;
}
.col-width-resizer:hover .resizer-handle-grip span,
.col-width-resizer.is-dragging .resizer-handle-grip span {
    background-color: #2563eb !important;
}

/* Prevent nested inputs/rows from horizontal clipping */
.col-manual-input .row,
.col-manual-input div[class*="row"],
.col-diagnostic-output .row,
.col-diagnostic-output div[class*="row"] {
    flex-wrap: wrap !important;
    gap: 8px !important;
}
.col-manual-input input,
.col-manual-input textarea,
.col-manual-input .gr-form,
.col-manual-input .gr-box,
.col-manual-input .gr-group,
.col-diagnostic-output .gr-form,
.col-diagnostic-output .gr-box,
.col-diagnostic-output .gr-group {
    min-width: 0 !important;
    max-width: 100% !important;
}

/* Sleek High-Contrast WebKit Scrollbars (Chrome, Safari, Edge) */
.col-sample-input::-webkit-scrollbar,
.col-manual-input::-webkit-scrollbar,
.col-diagnostic-output::-webkit-scrollbar {
    width: 9px !important;
    display: block !important;
}
.col-sample-input::-webkit-scrollbar-track,
.col-manual-input::-webkit-scrollbar-track,
.col-diagnostic-output::-webkit-scrollbar-track {
    background: #eff6ff !important;
    border-radius: 5px !important;
    border-left: 1px solid #bfdbfe !important;
}
.col-sample-input::-webkit-scrollbar-thumb,
.col-manual-input::-webkit-scrollbar-thumb,
.col-diagnostic-output::-webkit-scrollbar-thumb {
    background: #2563eb !important;
    border-radius: 5px !important;
}
.col-sample-input::-webkit-scrollbar-thumb:hover,
.col-manual-input::-webkit-scrollbar-thumb:hover,
.col-diagnostic-output::-webkit-scrollbar-thumb:hover {
    background: #1d4ed8 !important;
}

/* Responsive Fallback for Tablets (< 880px) */
@media (max-width: 879px) {
    body, .gradio-container {
        height: auto !important;
        max-height: none !important;
        overflow-y: auto !important;
    }
    .workstation-row-container {
        flex-direction: column !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        gap: 14px !important;
    }
    .col-width-resizer {
        display: none !important;
    }
    .col-sample-input,
    .col-manual-input,
    .col-diagnostic-output {
        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
        height: 600px !important;
        max-height: 600px !important;
        overflow-y: scroll !important;
    }
}

/* -------------------------------------------------------------------------
   4. PUNCHY TITLE BARS & ACCENTS
   ------------------------------------------------------------------------- */
.column-title-bar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 6px 10px !important;
    margin: -4px -4px 10px -4px !important;
    background: #eff6ff !important;
    border-left: 4px solid #2563eb !important;
    border-radius: 4px !important;
}
.column-title-text {
    font-size: 13.5px !important;
    font-weight: 800 !important;
    color: #1e40af !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.column-badge {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #1d4ed8 !important;
    background-color: #ffffff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 4px !important;
    padding: 2px 7px !important;
}

/* Specimen Selection Cards */
.specimen-preset-card {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 7px 9px !important;
    margin-bottom: 7px !important;
    transition: all 0.15s ease !important;
}
.specimen-preset-card:hover {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
    box-shadow: 0 2px 5px rgba(37, 99, 235, 0.1) !important;
}
.specimen-preset-card.active-selected {
    border-color: #2563eb !important;
    border-left: 4px solid #2563eb !important;
    background-color: #eff6ff !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

/* Specimen Patient Tag Chip */
.specimen-patient-tag {
    font-size: 10.5px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    display: inline-block !important;
    margin-top: 3px !important;
}

/* Punchy Status Badges */
.status-pill-malignant {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    background-color: #ffedd5 !important;
    color: #c2410c !important;
    border: 1.5px solid #fdba74 !important;
    padding: 2px 8px !important;
    border-radius: 12px !important;
    font-size: 11px !important;
    font-weight: 800 !important;
}
.status-pill-normal {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    background-color: #dcfce7 !important;
    color: #15803d !important;
    border: 1.5px solid #86efac !important;
    padding: 2px 8px !important;
    border-radius: 12px !important;
    font-size: 11px !important;
    font-weight: 800 !important;
}
.status-pill-benign {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    background-color: #fef3c7 !important;
    color: #b45309 !important;
    border: 1.5px solid #fde047 !important;
    padding: 2px 8px !important;
    border-radius: 12px !important;
    font-size: 11px !important;
    font-weight: 800 !important;
}

/* Input Group Framing */
.input-sub-section {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 9px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}
.input-sub-header {
    font-size: 11.5px !important;
    font-weight: 800 !important;
    color: #1e3a8a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
    margin-bottom: 6px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

/* Section Header matching User Cancer Diagnosis Specification */
.input-section-header {
    font-size: 12px !important;
    font-weight: 800 !important;
    color: #1e40af !important;
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-left: 3.5px solid #2563eb !important;
    padding: 5px 10px !important;
    border-radius: 4px !important;
    margin-bottom: 8px !important;
    letter-spacing: 0.3px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

/* Clinical Reference Range Chip */
.clinical-ref-chip {
    font-size: 10.5px !important;
    font-weight: 600 !important;
    color: #0369a1 !important;
    background: #e0f2fe !important;
    border: 1px solid #bae6fd !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    display: inline-block !important;
    margin-bottom: 4px !important;
}

/* Punchy Primary CTA Button (Blue Gradient with White Text) */
.btn-punchy-primary, button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 800 !important;
    font-size: 13.5px !important;
    padding: 9px 16px !important;
    box-shadow: 0 3px 8px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease !important;
}
.btn-punchy-primary:hover, button.primary:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Secondary Buttons (Crisp White with Punchy Blue Border) */
.btn-punchy-secondary, button.secondary {
    background: #ffffff !important;
    color: #1d4ed8 !important;
    border: 1.5px solid #2563eb !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    transition: all 0.15s ease !important;
}
.btn-punchy-secondary:hover, button.secondary:hover {
    background: #eff6ff !important;
    border-color: #1d4ed8 !important;
    color: #1e40af !important;
}

/* Punchy Radio Options (Force White/Light Palette, ZERO Dark Boxes) */
fieldset, .gr-radio, .gr-radio-group, div[data-testid="radio-group"],
.gradio-radio, div.radio-group,
.dark fieldset, .dark .gr-radio, .dark .gr-radio-group, .dark div[data-testid="radio-group"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    padding: 3px !important;
}
fieldset label, div[data-testid="radio-group"] label, .gradio-radio label,
.dark fieldset label, .dark div[data-testid="radio-group"] label {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
    margin: 2px !important;
    padding: 3px 7px !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 11.5px !important;
}
fieldset label:hover, div[data-testid="radio-group"] label:hover {
    border-color: #3b82f6 !important;
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
}
fieldset label.selected, fieldset label:has(input:checked),
div[data-testid="radio-group"] label.selected, div[data-testid="radio-group"] label:has(input:checked),
.dark fieldset label.selected, .dark fieldset label:has(input:checked) {
    background-color: #eff6ff !important;
    background: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
    font-weight: 800 !important;
    box-shadow: 0 0 0 1px #2563eb !important;
}

/* Inputs, Textareas, Sliders */
input, textarea, select {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
}

/* Viewport Container and Overlays */
.viewer-container {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    position: relative !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}
.viewer-slide-badge {
    position: absolute !important;
    top: 8px !important;
    left: 8px !important;
    z-index: 20 !important;
    background-color: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    padding: 4px 9px !important;
    border-radius: 4px !important;
    font-size: 11.5px !important;
    font-weight: 800 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08) !important;
}
.viewer-scale-bar {
    position: absolute !important;
    bottom: 8px !important;
    left: 8px !important;
    z-index: 20 !important;
    background-color: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid #cbd5e1 !important;
    padding: 3px 8px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}
.viewer-timestamp {
    position: absolute !important;
    bottom: 8px !important;
    right: 8px !important;
    z-index: 20 !important;
    background-color: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid #cbd5e1 !important;
    color: #475569 !important;
    padding: 3px 8px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-family: monospace !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}

/* Accordions in Column 3 */
.gr-accordion {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    margin-bottom: 8px !important;
}
.gr-accordion-header {
    background: #eff6ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    font-size: 12.5px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding: 6px 10px !important;
}

/* -------------------------------------------------------------------------
   5. CANCER DIAGNOSIS SYSTEM REPORT CARD (EXACT USER SPECIFICATION)
   ------------------------------------------------------------------------- */
.cancer-diag-card {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
}

/* Header */
.diag-card-header {
    background: linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%) !important;
    border-bottom: 2.5px solid #2563eb !important;
    padding: 10px 14px !important;
}
.diag-card-top-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
}
.diag-card-brand-title {
    font-size: 18px !important;
    font-weight: 900 !important;
    color: #1e3a8a !important;
    letter-spacing: -0.3px !important;
}
.diag-card-pid-badge {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    color: #1e40af !important;
    background: #eff6ff !important;
    border: 1.5px solid #93c5fd !important;
    padding: 3px 10px !important;
    border-radius: 6px !important;
}
.diag-card-subtitle {
    font-size: 11px !important;
    font-weight: 800 !important;
    color: #64748b !important;
    letter-spacing: 0.6px !important;
    margin-top: 3px !important;
    text-transform: uppercase !important;
}

/* Section Framing */
.diag-section {
    padding: 12px 14px !important;
    border-bottom: 1.5px solid #e2e8f0 !important;
}
.diag-section-header {
    font-size: 13px !important;
    font-weight: 800 !important;
    color: #1e40af !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
    margin-bottom: 8px !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

/* 01. Classification */
.diag-cancer-box {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 4px solid #2563eb !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    margin-bottom: 10px !important;
}
.diag-cancer-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 4px !important;
}
.diag-cancer-name {
    font-size: 16px !important;
    font-weight: 900 !important;
    color: #0f172a !important;
}
.diag-cancer-conf {
    font-size: 14.5px !important;
    font-weight: 800 !important;
    color: #2563eb !important;
}

.diag-prob-label {
    font-size: 11px !important;
    font-weight: 800 !important;
    color: #64748b !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    margin: 8px 0 6px 0 !important;
}
.diag-prob-row {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 6px !important;
    font-size: 12px !important;
}
.diag-prob-name {
    flex: 0 0 145px !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
.diag-prob-track {
    flex: 1 !important;
    background: #e2e8f0 !important;
    height: 10px !important;
    border-radius: 5px !important;
    overflow: hidden !important;
}
.diag-prob-fill {
    height: 100% !important;
    border-radius: 5px !important;
    transition: width 0.3s ease !important;
}
.diag-prob-pct {
    flex: 0 0 55px !important;
    text-align: right !important;
    font-weight: 700 !important;
    font-size: 11.5px !important;
    color: #0f172a !important;
}

/* 02. TNM Staging */
.diag-tnm-table {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    margin-bottom: 8px !important;
    font-size: 12.5px !important;
}
.diag-tnm-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 3px 0 !important;
    border-bottom: 1px dashed #f1f5f9 !important;
}
.diag-tnm-key {
    color: #475569 !important;
    font-weight: 600 !important;
}
.diag-tnm-val {
    font-family: ui-monospace, monospace !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    background: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 1px solid #bfdbfe !important;
    padding: 1px 8px !important;
    border-radius: 4px !important;
}
.diag-overall-stage {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    margin: 8px 0 6px 0 !important;
    font-size: 13px !important;
}
.diag-overall-stage strong {
    color: #1d4ed8 !important;
    font-weight: 800 !important;
    font-size: 14.5px !important;
}

/* Interactive Accordion Details */
details.diag-details {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    margin-top: 6px !important;
    font-size: 12px !important;
}
details.diag-details summary {
    cursor: pointer !important;
    color: #2563eb !important;
    font-weight: 700 !important;
    padding: 6px 10px !important;
    user-select: none !important;
}
details.diag-details summary:hover {
    color: #1d4ed8 !important;
    background: #eff6ff !important;
}
.diag-details-body {
    padding: 8px 10px !important;
    background: #f8fafc !important;
    border-top: 1px solid #e2e8f0 !important;
    color: #334155 !important;
    font-size: 11.5px !important;
    line-height: 1.5 !important;
}

/* 03. Survival Estimation */
.diag-survival-row {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 8px !important;
}
.diag-survival-num {
    font-size: 18px !important;
    font-weight: 900 !important;
}
.diag-survival-track {
    background: #e2e8f0 !important;
    height: 12px !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}
.diag-survival-fill {
    height: 100% !important;
    border-radius: 6px !important;
    transition: width 0.4s ease !important;
}
.diag-survival-ticks {
    display: flex !important;
    justify-content: space-between !important;
    font-size: 10px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    margin-top: 3px !important;
    margin-bottom: 8px !important;
}

/* 04. Treatment Recommendations */
.diag-treatment-title {
    font-size: 11.5px !important;
    font-weight: 800 !important;
    color: #475569 !important;
    letter-spacing: 0.4px !important;
    margin-bottom: 6px !important;
}
.diag-treatment-list {
    display: flex !important;
    flex-direction: column !important;
    gap: 7px !important;
    margin-bottom: 8px !important;
}
.diag-treatment-card {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 3.5px solid #2563eb !important;
    padding: 7px 10px !important;
    border-radius: 5px !important;
}
.diag-treatment-heading {
    font-size: 12.5px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin-bottom: 2px !important;
}
.diag-treatment-desc {
    font-size: 11px !important;
    color: #475569 !important;
    line-height: 1.4 !important;
}
.diag-treatment-warning {
    background: #fffbeb !important;
    border: 1.5px solid #fde68a !important;
    border-radius: 6px !important;
    padding: 7px 10px !important;
    color: #92400e !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    margin-top: 8px !important;
}

/* 06. Model Information & Footer */
.diag-model-specs {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}
.diag-model-specs strong {
    color: #1e40af !important;
}
.diag-disclaimer-box {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 3px solid #f59e0b !important;
    padding: 6px 10px !important;
    border-radius: 4px !important;
    font-size: 10.5px !important;
    font-weight: 600 !important;
    color: #64748b !important;
    margin-top: 8px !important;
    line-height: 1.4 !important;
}
"""

AISIGHT_CUSTOM_CSS = GCAEMMCD_CUSTOM_CSS


def get_gcaemmcd_theme():
    """Builds a pure light Gradio theme where dark variables are neutralized."""
    return gr.themes.Default(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate"
    ).set(
        body_background_fill="#f8fafc",
        body_background_fill_dark="#f8fafc",
        body_text_color="#0f172a",
        body_text_color_dark="#0f172a",
        body_text_color_subdued="#334155",
        body_text_color_subdued_dark="#334155",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#f1f5f9",
        background_fill_secondary_dark="#f1f5f9",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_label_text_color="#0f172a",
        block_label_text_color_dark="#0f172a",
        block_title_text_color="#0f172a",
        block_title_text_color_dark="#0f172a",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#cbd5e1",
        input_border_color_dark="#cbd5e1",
        checkbox_label_background_fill="#ffffff",
        checkbox_label_background_fill_dark="#ffffff",
        checkbox_label_background_fill_selected="#eff6ff",
        checkbox_label_background_fill_selected_dark="#eff6ff",
        checkbox_label_text_color="#0f172a",
        checkbox_label_text_color_dark="#0f172a",
        checkbox_label_text_color_selected="#1d4ed8",
        checkbox_label_text_color_selected_dark="#1d4ed8",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_text_color="#1d4ed8",
        button_secondary_text_color_dark="#1d4ed8",
        border_color_primary="#e2e8f0",
        border_color_primary_dark="#e2e8f0"
    )


def get_top_navbar_html():
    return """
    <div class="gca-top-bar">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div class="gca-brand">
                <span class="gca-logo-icon">🧬</span>
                <span>GCAEMMCD<span style="font-size: 12.5px; font-weight: 700; color: #64748b; margin-left: 8px;">Clinical Pathology Workstation</span></span>
            </div>
            <div class="gca-engine-badge">
                <span>Architecture:</span>
                <strong>Tri-Modal GCA (v2.4)</strong>
            </div>
        </div>
        <div class="gca-telemetry">
            <div><span>Hardware:</span> <strong style="color: #1d4ed8;">NVIDIA RTX 2050 (CUDA)</strong></div>
            <div><span>Backbones:</span> <strong>ResNet-50 · Phikon ViT-B · BioBERT</strong></div>
            <span class="gca-status-pill-active">● Models Calibrated</span>
        </div>
    </div>
    """


def get_sub_header_html(specimen_id="Specimen S-01", status="Ready", patient_id="PID-84021"):
    return f"""
    <div class="gca-sub-header">
        <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
            <div style="font-weight: 800; color: #0f172a;">
                Diagnostic Session › <span class="gca-meta-pill">{specimen_id}</span>
                <span style="margin-left: 8px; font-size: 11.5px; color: #1e40af; background: #eff6ff; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 4px;">Patient: <strong>{patient_id}</strong></span>
            </div>
            <div><span style="color: #64748b;">Active Modalities:</span> <strong style="color: #1e40af;">H&E Tile + Clinical Narrative + 16 Biomarkers</strong></div>
            <div><span style="color: #64748b;">Pipeline Status:</span> <span style="color: #15803d; font-weight: 800;">● {status}</span></div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #059669; font-weight: 700; font-size: 12px;">● Optical Domain & OOD Protection: <strong>ACTIVE</strong></span>
        </div>
    </div>
    """


def get_viewer_toolbar_html(specimen_id="Specimen S-01"):
    return f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; margin-bottom: 8px; font-size: 12px;">
        <div>
            <strong style="color: #1e40af;">Pathology Micrograph Viewport:</strong>
            <span style="color: #2563eb; font-weight: 800; margin-left: 4px;">{specimen_id}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="color: #64748b;">Field of View: <strong>1.84 mm²</strong></span>
            <span style="color: #059669; font-weight: 800;">● Stage 0 Optical Check: Validated</span>
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
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; color: #64748b; text-align: center; padding: 4px 14px; font-size: 11px; font-weight: 500; height: 26px; line-height: 16px; box-sizing: border-box; margin-top: 4px;">
        GCAEMMCD Multimodal Cancer Diagnosis Decision Support System • For Investigational & Research Use Only • Zero-Leakage 5-Fold Cross-Validated
    </div>
    """
