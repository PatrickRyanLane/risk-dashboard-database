import { mountParityPage, prefetchParityPage } from '../../parity-runtime.js';

const page = {
  "styles": ":root {\n  --bg: #044152;\n  --card: #092e37;\n  --muted: #a2ebf3;\n  --text: #ebf2f2;\n  --accent: #58dbed;\n  --black: #1f2121;\n  \n  --pill-neg: #ff8261;\n  --pill-neu: #cfdbdd;\n  --pill-pos: #82c618;\n\n  --stroke: #0e2230;\n  --chip: #12343e;\n  --chipBorder: #174550;\n  \n  --stock-pos:#82c616; \n  --stock-neg:#ff8261;\n}\n\n*{box-sizing:border-box}\nbody{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif}\n.wrap{max-width:1200px;margin:0 auto;padding:20px}\nh1{font-size:28px;margin:6px 0 18px}\n\n/* Tooltips */\n.tooltip-header {\n  position: relative;\n  display: inline-block;\n  border-bottom: 1px dotted #a2ebf3;\n  cursor: help;\n}\n\n.tooltip-header .tooltip-text {\n  visibility: hidden;\n  width: 240px;\n  background-color: #1f2121;\n  color: #ebf2f2;\n  text-align: left;\n  border: 1px solid #2a3b5e;\n  border-radius: 8px;\n  padding: 10px 12px;\n  position: absolute;\n  z-index: 1000;\n  top: 125%;\n  left: 50%;\n  margin-left: -120px;\n  opacity: 0;\n  transition: opacity 0.3s ease;\n  font-size: 12px;\n  line-height: 1.5;\n  font-weight: normal;\n  white-space: normal;\n  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);\n}\n\n.tooltip-header .tooltip-text::after {\n  content: \"\";\n  position: absolute;\n  bottom: 100%;\n  left: 50%;\n  margin-left: -5px;\n  border-width: 5px;\n  border-style: solid;\n  border-color: transparent transparent #1f2121 transparent;\n}\n\n.tooltip-header:hover .tooltip-text {\n  visibility: visible;\n  opacity: 1;\n}\n\n@media (max-width: 768px) {\n  .tooltip-header .tooltip-text {\n    width: 200px;\n    margin-left: -100px;\n    font-size: 11px;\n  }\n}\n\n/* Controls */\n.controls{display:flex;flex-direction:column;gap:8px;margin:0 0 16px}\n.controls select,\n.controls input,\n.controls button{height:32px;padding:6px 10px;font-size:12px;line-height:1.2;border-radius:8px}\n.controls .controls-grid{\n  display:grid;\n  grid-template-columns:170px 220px 190px 190px auto auto;\n  gap:8px 12px;\n  align-items:end;\n  width:100%;\n}\n.controls-row{display:flex;justify-content:flex-end;margin:0 0 8px}\n.refresh-panel{\n  display:flex;\n  align-items:center;\n  justify-content:center;\n  gap:0;\n  padding:8px 10px;\n  border:1px solid var(--stroke);\n  border-radius:10px;\n  background:var(--card);\n}\n.refresh-panel button{height:32px;padding:6px 10px;font-size:12px;line-height:1.2;border-radius:8px}\n.controls .field-label{font-size:12px;margin:0;padding-left:8px;line-height:1.2}\n.controls .field-label-empty{padding-left:0}\n.controls .controls-inputs{align-items:flex-start}\n.controls .controls-inputs input,\n.controls .controls-inputs select{width:100%;min-width:0}\n.controls .controls-inputs button{width:auto;white-space:nowrap;justify-self:start}\n.controls .date-input-stack{display:flex;flex-direction:column;gap:8px;min-width:0}\n.controls .lookback-controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;width:100%}\n.controls .lookback-btn{min-width:0;padding:0 6px;font-weight:600}\n.controls .lookback-btn.active{border-color:#58dbed;box-shadow:0 0 0 1px rgba(88,219,237,.35);color:#58dbed}\n.controls .controls-aux{display:flex;flex-wrap:wrap;align-items:center;gap:8px}\n.filter-toggle.active{border-color:#58dbed;box-shadow:0 0 0 1px rgba(88,219,237,.35);color:#58dbed}\n.crisis-toggle{border-color:#7b1d1d;color:#ffb3b3}\n.crisis-toggle.active{border-color:#ff4545;box-shadow:0 0 0 1px rgba(255,69,69,.4);color:#ff5c5c}\n.controls .spacer{display:none}\n@media (max-width: 980px){\n  .controls-row{justify-content:stretch}\n  .refresh-panel{width:100%;justify-content:center}\n  .controls .controls-header{display:none}\n  .controls .controls-grid{grid-template-columns:1fr}\n  .controls .controls-inputs button{width:100%}\n}\n.card{background:var(--card);border:1px solid var(--stroke);border-radius:12px;padding:12px 12px}\nselect,input[type=\"text\"]{background:#1f2121;border:1px solid #23314d;border-radius:10px;color:var(--text);padding:8px 10px}\nbutton{background:#1f2121;border:1px solid #2a3b5e;color:var(--text);border-radius:10px;padding:8px 12px;cursor:pointer}\nbutton:hover{filter:brightness(1.08)}\n\n.grid{\n  display:grid;\n  grid-auto-rows:auto;\n  gap:16px;\n  margin:16px 0;\n}\n\n@media (min-width:980px){\n  .grid{\n    grid-auto-rows:auto;\n    gap:16px;\n  }\n}\n\n.chart-card{ background:var(--card); border:1px solid var(--stroke);\n  border-radius:12px; padding:16px; position:relative; min-height:400px; height:48vh }\n\n.chart-card canvas{\n  width:100% !important;\n  height:100% !important;\n  display:block;\n}\n.chart-card[data-chart=\"composite\"]{ min-height:460px; height:58vh }\n.chart-card.loading::after{content:\"\";position:absolute;left:16px;right:16px;top:58px;bottom:16px;border-radius:10px;background:linear-gradient(90deg,rgba(255,255,255,.05),rgba(255,255,255,.18),rgba(255,255,255,.05));background-size:200% 100%;animation:skeleton 1.2s ease-in-out infinite;pointer-events:none}\n.chart-card.loading canvas{opacity:.2}\n.chart-card[draggable=\"true\"]{cursor:grab}\n.chart-card h3{margin:0}\n.chart-card[draggable=\"true\"] h3{\n  display:flex;\n  align-items:flex-start;\n  gap:10px;\n  padding-left:2px;\n}\n.chart-card[draggable=\"true\"] h3 .drag-handle{\n  display:inline-flex;\n  align-items:center;\n  justify-content:center;\n  font-size:25px;\n  line-height:.7;\n  color:rgba(235,242,242,.38);\n  text-shadow:0 1px 0 rgba(0,0,0,.55), 0 -1px 0 rgba(255,255,255,.06);\n  opacity:.55;\n  transform:translateY(1px);\n  cursor:grab;\n}\n.chart-card[draggable=\"true\"] h3 .drag-handle::before{content:\"\u2630\"}\n.chart-card.dragging{opacity:.55}\n.chart-card.drag-over{outline:1px dashed rgba(88,219,237,.7);outline-offset:2px}\n\n.chart-card.chart-collapsed{grid-column:1 / -1;padding:16px;min-height:0;height:auto}\n.chart-card.chart-collapsed canvas{display:none !important}\n.chart-card.chart-collapsed .chart-actions{\n  display:flex;\n  top:16px;\n  right:16px;\n}\n.chart-card.chart-collapsed .chart-actions .dates-pager{display:none}\n.chart-card.chart-collapsed.loading::after{display:none}\n.chart-card .chart-collapsible-title{cursor:pointer;user-select:none}\n\n.chart-actions{\n  position:absolute;\n  top:16px; right:16px;\n  display:flex; align-items:center; gap:10px;\n}\n.chart-actions .dates-range{ font-size:12px; }\n.chart-actions .dates-pager{ display:flex; gap:6px; }\n.chart-actions .dates-pager button{\n  background:#1f2121; border:1px solid #2a3b5e; color:var(--text);\n  border-radius:8px; padding:4px 8px; cursor:pointer;\n}\n.chart-actions .dates-pager button[disabled]{ opacity:.45; cursor:not-allowed; }\n\n/* Table */\n.table-card{background:var(--card);border:1px solid var(--stroke);border-radius:12px;padding:10px}\n.table-hint{margin:0 0 8px;color:var(--muted)}\n.load-status{margin-top:10px;width:100%;display:flex;flex-direction:column;align-items:center}\n.load-bar{height:6px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden}\n.load-bar-fill{height:100%;width:0%;background:var(--accent);transition:width .2s ease}\n.load-items{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;font-size:12px;color:var(--muted);justify-content:center;text-align:center}\n.load-pill{padding:2px 8px;border-radius:999px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.1)}\n.load-pill.done{color:#82c618;border-color:rgba(130,198,24,.35);background:rgba(130,198,24,.1)}\n.load-pill.error{color:#ff8261;border-color:rgba(255,130,97,.35);background:rgba(255,130,97,.1)}\ntable{width:100%;border-collapse:separate;border-spacing:0 6px;table-layout:auto}\nthead th{font-size:11px;color:var(--muted);text-align:center;padding:6px 6px;vertical-align:middle;line-height:1.2;white-space:nowrap}\nthead th:first-child{white-space:normal;min-width:140px}\ntbody td{padding:8px 6px;background:#1f2121;border-top:1px solid #0f1831;border-bottom:1px solid #0f1831;text-align:center;white-space:nowrap}\ntbody td:first-child{text-align:left;white-space:normal}\nthead th.action-col, tbody td.action-col{width:88px;min-width:88px;max-width:88px}\ntbody td.action-col button{min-width:62px}\ntbody tr{transition:background-color 0.15s}\ntbody tr:hover{background:rgba(88,219,237,.08)}\ntbody tr td:first-child{border-left:1px solid #0f1831;border-radius:10px 0 0 10px}\ntbody tr td:last-child{border-right:1px solid #0f1831;border-radius:0 10px 10px 0}\n.skeleton-row td{padding:10px 12px}\n.skeleton-bar{height:12px;border-radius:8px;background:linear-gradient(90deg,rgba(255,255,255,.05),rgba(255,255,255,.2),rgba(255,255,255,.05));background-size:200% 100%;animation:skeleton 1.2s ease-in-out infinite}\n.skeleton-bar.full{width:100%;height:16px}\n.skeleton-bar.wide{width:90%}\n.skeleton-bar.med{width:60%}\n.skeleton-bar.narrow{width:35%}\n@keyframes skeleton{0%{background-position:200% 0}100%{background-position:-200% 0}}\n\n.ceo-cell-content{display:flex;align-items:center;gap:8px}\n.ceo-name-block{display:flex;flex-direction:column;gap:2px}\n.ceo-name{font-weight:500}\n.company-subheader{font-size:11px;color:#7a9ca3;font-weight:400}\n .fav-inline{background:transparent;border:0;color:#666;font-size:14px;line-height:1;padding:0 4px;display:inline-flex;align-items:center;justify-content:center;cursor:default}\n .fav-inline.active{color:#ffd54f}\n\n.stock-price{font-family:'Courier New',monospace;font-weight:500;color:var(--text)}\n.stock-change{font-weight:600}\n.stock-change.positive{color:var(--stock-pos)}\n.stock-change.negative{color:var(--stock-neg)}\n\n.sparkline-cell{text-align:center;padding:6px 8px !important;cursor:pointer}\n.sparkline-cell:hover{background:rgba(88,219,237,.15)}\n.sparkline{display:inline-block;vertical-align:middle}\n\n.pill{font-size:12px;border-radius:999px;padding:3px 8px;border:1px solid var(--chipBorder);background:var(--chip);color:#cfe0ff}\n.pill.low  { background:rgba(130,198,24,.15); color:#82c618; }\n.pill.med  { background:rgba(207,219,221,.25); color:#cfdbdd; }\n.pill.high { background:rgba(255,130,97,.15); color:#ff8261; }\n.link{color:var(--accent);text-decoration:underline}\n\n.table-scroll{\n  overflow-x:auto;\n  -webkit-overflow-scrolling:touch;\n}\n\n.table-scroll::-webkit-scrollbar{height:8px}\n.table-scroll::-webkit-scrollbar-track{background:rgba(255,255,255,.05);border-radius:8px}\n.table-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:8px}\n\n@media (max-width: 960px){\n  .table-card{ padding:8px }\n  .table-hint{ font-size:12px }\n  thead th, tbody td{ padding:6px 4px }\n  .pill{ font-size:11px; padding:3px 8px }\n  .pagination{ gap:6px; flex-wrap:wrap }\n}\n\ntbody td:first-child,\nthead th:first-child{\n  position: sticky;\n  left: 0;\n  z-index: 2;\n  white-space: normal;\n  box-shadow: 2px 0 0 0 rgba(255,255,255,.06);\n  background: var(--card);\n}\ntbody td:first-child{\n  background: var(--black);\n}\n\n.pagination{display:flex;gap:8px;align-items:center;justify-content:flex-end;margin-top:8px}\n\n.modal{\n  position:fixed;\n  inset:0;\n  background:rgba(0,0,0,.55);\n  display:none;\n  align-items:center;\n  justify-content:center;\n  padding:24px;\n  z-index:10000;\n}\n.modal.open{ display:flex; }\n\n.modal .box{\n  max-width:900px; width:100%;\n  max-height:85vh; overflow:auto;\n  background:#0e152a; border:1px solid #1b2745; border-radius:16px;\n  position:relative;\n  z-index:10001;\n}\n.modal header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #12203d}\n.modal header h3{margin:0;font-size:16px}\n.modal .content{padding:16px}\n.modal-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}\n.modal-tab{\n  background:#1f2121;\n  border:1px solid #2a3b5e;\n  color:var(--muted);\n  border-radius:999px;\n  padding:6px 12px;\n  font-size:12px;\n  line-height:1.2;\n  cursor:pointer;\n}\n.modal-tab.active{\n  border-color:rgba(88,219,237,.55);\n  background:rgba(88,219,237,.12);\n  color:#58dbed;\n}\n.modal-tab-panel{min-width:0}\n.modal-toolbar{\n  display:flex;\n  align-items:center;\n  justify-content:space-between;\n  gap:12px;\n  flex-wrap:nowrap;\n  margin:0 0 12px;\n}\n.modal-toolbar .modal-tabs{\n  margin:0;\n  flex:1 1 auto;\n  min-width:0;\n  flex-wrap:nowrap;\n  overflow-x:auto;\n}\n.modal-date-panel{\n  display:flex;\n  flex-direction:column;\n  align-items:flex-end;\n  gap:6px;\n  min-width:190px;\n  padding:8px 10px;\n  border:1px solid #23314d;\n  border-radius:10px;\n  background:rgba(31,33,33,.5);\n}\n.modal-date-panel label{font-size:12px;color:var(--muted);width:100%;text-align:left;padding-left:3px}\n.modal-date-panel select{width:100%;min-width:150px}\n.modal-date-panel button{width:100%}\n.modal-date-panel .modal-date-status{font-size:11px;color:var(--muted);text-align:left;width:100%;padding-left:3px}\n.close{color:#aaa;float:right;font-size:28px;font-weight:bold;cursor:pointer;line-height:20px}\n.close:hover,.close:focus{color:var(--text)}\n.badges{display:flex;gap:8px;margin-top:8px}\n.badge{font-size:12px;padding:4px 8px;border-radius:999px;border:1px solid transparent}\n.badge.positive{\n  background:rgba(130,198,24,.15);\n  color:#82c618;\n  border-color:rgba(130,198,24,.35);\n}\n.badge.neutral{\n  background:rgba(207,219,221,.25);\n  color:#cfdbdd;\n  border-color:rgba(207,219,221,.45);\n}\n.badge.negative{\n  background:rgba(255,130,97,.15);\n  color:#ff8261;\n  border-color:rgba(255,130,97,.35);\n}\n.badge.controlled{\n  background:rgba(88,219,237,.15);\n  color:#58dbed;\n  border-color:rgba(88,219,237,.35);\n}\n.badge.uncontrolled{\n  background:rgba(255,130,97,.15);\n  color:#ff8261;\n  border-color:rgba(255,130,97,.35);\n}\n.serp-flags{margin-top:6px;font-size:12px;opacity:.8;letter-spacing:.2px}\n.edit-flags{display:inline-flex;gap:10px;align-items:center;font-size:12px;opacity:.85;white-space:nowrap}\n.muted{color:var(--muted)}\n.toggle-switch{display:inline-flex;align-items:center;gap:8px;cursor:pointer}\n.toggle-switch input{display:none}\n.toggle-slider{position:relative;width:36px;height:20px;background:#1f2b4a;border:1px solid #2b3d63;border-radius:999px;transition:all .2s ease}\n.toggle-slider::after{content:'';position:absolute;top:2px;left:2px;width:14px;height:14px;background:#8aa0b6;border-radius:50%;transition:transform .2s ease, background .2s ease}\n.toggle-switch input:checked + .toggle-slider{background:rgba(88,219,237,.2);border-color:rgba(88,219,237,.5)}\n.toggle-switch input:checked + .toggle-slider::after{transform:translateX(16px);background:#58dbed}\n.toggle-text{font-size:12px;color:#cfdbdd}\n\n#stockChartModal .box{max-width:1000px; width:95%; max-height:90vh; overflow:auto;}\n#stockChart{width:100% !important; height:450px !important; min-height:400px;}\n\n.serp-cards{display:grid;gap:12px;margin-top:12px}\n.serp-card{background:var(--card);border:1px solid var(--stroke);border-radius:12px;padding:14px 16px;position:relative}\n.serp-feature-flag{\n  position:absolute;\n  top:10px;\n  right:12px;\n  font-size:11px;\n  text-transform:uppercase;\n  letter-spacing:.3px;\n  padding:3px 8px;\n  border-radius:999px;\n  background:rgba(88,219,237,.12);\n  border:1px solid rgba(88,219,237,.35);\n  color:#58dbed;\n}\n.serp-domain{font-size:12px;color:var(--muted);margin-bottom:4px}\n.serp-title a{color:var(--text);text-decoration:underline}\n.serp-snippet{color:var(--muted);margin-top:6px;line-height:1.35}\n.serp-metrics{margin:4px 0 0;color:var(--muted)}\n.serp-metrics b{color:var(--text)}\n.serp-feature-viz{display:flex;gap:20px;flex-wrap:wrap;align-items:stretch;margin:12px 0}\n.serp-feature-viz > div{min-width:0}\n.serp-card.serp-feature-summary{\n  margin-top:10px;\n  max-width:100%;\n  min-width:0;\n  background:rgba(120,82,255,.18);\n  border-color:rgba(170,140,255,.55);\n}\n.serp-card.serp-feature-summary .serp-snippet{margin-top:0;overflow-wrap:anywhere;word-break:break-word;color:#eee6ff}\n.serp-card.serp-feature-summary .ghost-btn{align-self:center}\n@media (max-width: 900px){\n  .modal-toolbar{flex-direction:column}\n  .modal-toolbar .modal-tabs{width:100%;flex:1 1 auto}\n  .modal-date-panel{width:100%;align-items:stretch}\n  .modal-date-panel .modal-date-status{text-align:left}\n  .serp-feature-viz{flex-direction:column}\n  .serp-feature-viz > div{height:200px}\n}\n\n#stockChartModal header {\n  display: flex;\n  justify-content: space-between;\n  align-items: center;\n  padding: 14px 18px;\n  border-bottom: 1px solid #12203d;\n  gap: 12px;\n}\n\n#stockChartModal header h3 {\n  margin: 0;\n  font-size: 16px;\n  line-height: 1.3;\n  flex: 1;\n  padding-right: 12px;\n}\n\n/* Volume toggle button */\n.volume-toggle {\n  background: rgba(88, 219, 237, 0.15);\n  border: 1px solid rgba(88, 219, 237, 0.3);\n  color: var(--accent);\n  border-radius: 8px;\n  padding: 6px 12px;\n  font-size: 13px;\n  cursor: pointer;\n  transition: all 0.2s ease;\n  white-space: nowrap;\n  font-weight: 500;\n}\n\n.volume-toggle:hover {\n  background: rgba(88, 219, 237, 0.25);\n  border-color: rgba(88, 219, 237, 0.5);\n  transform: translateY(-1px);\n}\n\n.volume-toggle.active {\n  background: var(--accent);\n  color: var(--black);\n  border-color: var(--accent);\n}\n\n.volume-toggle.active:hover {\n  background: #6de5f5;\n  border-color: #6de5f5;\n}\n\n/* Trends toggle button */\n.trends-toggle {\n  background: rgba(168, 85, 247, 0.15);\n  border: 1px solid rgba(168, 85, 247, 0.3);\n  color: #a855f7;\n  border-radius: 8px;\n  padding: 6px 12px;\n  font-size: 13px;\n  cursor: pointer;\n  transition: all 0.2s ease;\n  white-space: nowrap;\n  font-weight: 500;\n}\n\n.trends-toggle:hover {\n  background: rgba(168, 85, 247, 0.25);\n  border-color: rgba(168, 85, 247, 0.5);\n  transform: translateY(-1px);\n}\n\n.trends-toggle.active {\n  background: #a855f7;\n  color: white;\n  border-color: #a855f7;\n}\n\n.trends-toggle.active:hover {\n  background: #b975f7;\n  border-color: #b975f7;\n}\n\n\n#stockChartModal .content {\n  padding: 18px;\n}\n\n.sparkline-cell {\n  cursor: pointer;\n  transition: background-color 0.2s ease;\n  position: relative;\n}\n\n.sparkline-cell:hover {\n  background: rgba(88, 219, 237, 0.2) !important;\n}\n\n.sparkline-cell::after {\n  content: '';\n  position: absolute;\n  inset: 0;\n  border: 2px solid transparent;\n  border-radius: 8px;\n  pointer-events: none;\n  transition: border-color 0.2s ease;\n}\n\n.sparkline-cell:hover::after {\n  border-color: rgba(88, 219, 237, 0.4);\n}\n\n.sparkline-cell:hover .sparkline {\n  filter: brightness(1.2);\n}\n\n.stock-chart-loading {\n  display: flex;\n  align-items: center;\n  justify-content: center;\n  height: 400px;\n  color: var(--muted);\n  font-size: 14px;\n}\n\n.stock-chart-loading::before {\n  content: '\ud83d\udcca';\n  font-size: 32px;\n  margin-right: 12px;\n  animation: pulse 1.5s ease-in-out infinite;\n}\n\n@keyframes pulse {\n  0%, 100% { opacity: 0.5; transform: scale(1); }\n  50% { opacity: 1; transform: scale(1.1); }\n}\n\n@keyframes chartFadeIn {\n  from {\n    opacity: 0;\n    transform: translateY(10px);\n  }\n  to {\n    opacity: 1;\n    transform: translateY(0);\n  }\n}\n\n#stockChart {\n  animation: chartFadeIn 0.3s ease-out;\n}\n\n#stockChartModal .close {\n  transition: all 0.2s ease;\n  cursor: pointer;\n  user-select: none;\n  font-size: 32px;\n  line-height: 28px;\n  font-weight: 300;\n  padding: 0 4px;\n  margin: -4px 0;\n}\n\n#stockChartModal .close:hover {\n  color: var(--accent) !important;\n  transform: scale(1.15);\n}\n\n#stockChartModal .close:active {\n  transform: scale(0.95);\n}\n\n.sparkline {\n  transition: opacity 0.2s ease;\n}\n\n.sparkline-cell:active .sparkline {\n  opacity: 0.7;\n}\n\n@media (max-width: 1024px) {\n  #stockChartModal .box {\n    width: 92%;\n    max-width: 900px;\n  }\n  \n  #stockChart {\n    height: 420px !important;\n    min-height: 380px;\n  }\n  \n  #stockChartModal header h3 {\n    font-size: 15px;\n  }\n  \n  #stockChartModal .content {\n    padding: 16px;\n  }\n}\n\n@media (max-width: 768px) {\n  #stockChartModal .box {\n    width: 100%;\n    height: 100%;\n    max-width: 100%;\n    max-height: 100vh;\n    margin: 0;\n    border-radius: 0;\n    display: flex;\n    flex-direction: column;\n  }\n  \n  #stockChartModal header {\n    padding: 12px 16px;\n    flex-shrink: 0;\n  }\n  \n  #stockChartModal header h3 {\n    font-size: 14px;\n    line-height: 1.4;\n  }\n  \n  #stockChartModal .close {\n    font-size: 28px;\n    line-height: 24px;\n  }\n  \n  #stockChartModal .content {\n    padding: 12px 16px 16px;\n    overflow-y: auto;\n    flex: 1;\n    -webkit-overflow-scrolling: touch;\n  }\n  \n  #stockChart {\n    height: 350px !important;\n    min-height: 300px;\n    max-height: 60vh;\n  }\n  \n  #stockChartModal .close {\n    min-width: 44px;\n    min-height: 44px;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n  }\n}\n\n@media (max-width: 480px) {\n  #stockChartModal header h3 {\n    font-size: 13px;\n  }\n  \n  #stockChart {\n    height: 320px !important;\n    min-height: 280px;\n  }\n  \n  #stockChartModal .content {\n    padding: 10px 12px 14px;\n  }\n}\n\n@media (max-width: 768px) and (orientation: landscape) {\n  #stockChart {\n    height: 70vh !important;\n    min-height: 250px;\n  }\n}\n\n@media (min-width: 1400px) {\n  #stockChartModal .box {\n    max-width: 1200px;\n  }\n  \n  #stockChart {\n    height: 500px !important;\n  }\n  \n  #stockChartModal header h3 {\n    font-size: 18px;\n  }\n}\n\nbody:has(#stockChartModal.open) {\n  overflow: hidden;\n}\n\n@media (max-width: 768px) {\n  .sparkline-cell {\n    padding: 10px 8px !important;\n  }\n  \n  button {\n    min-height: 44px;\n    padding: 10px 14px;\n  }\n}\n\n@media (min-width: 769px) {\n  #stockChartModal {\n    backdrop-filter: blur(2px);\n    -webkit-backdrop-filter: blur(2px);\n  }\n}\n\n/* ===== ENHANCED HEATMAP LEGEND STYLES ===== */\n.heatmap-legend {\n  display: flex;\n  gap: 18px;\n  align-items: center;\n  justify-content: center;\n  padding: 10px 16px;\n  background: rgba(30, 40, 55, 0.7);\n  border: 1px solid rgba(88, 219, 237, 0.2);\n  border-radius: 8px;\n  margin-top: 14px;\n  font-size: 13px;\n  font-weight: 500;\n  flex-wrap: wrap;\n  color: #e3f2f7;\n}\n\n.heatmap-legend-item {\n  display: flex;\n  align-items: center;\n  gap: 8px;\n}\n\n.heatmap-legend-dot {\n  width: 12px;\n  height: 12px;\n  border-radius: 50%;\n  border: 2px solid #fff;\n  box-shadow: 0 2px 4px rgba(0,0,0,0.3);\n  flex-shrink: 0;\n}\n\n.heatmap-legend-dot.medium {\n  background: #ffaa44;\n}\n\n.heatmap-legend-dot.high {\n  background: #ff4444;\n}\n\n@media (max-width: 768px) {\n  .heatmap-legend {\n    font-size: 12px;\n    gap: 14px;\n    padding: 8px 12px;\n  }\n  \n  .heatmap-legend-dot {\n    width: 10px;\n    height: 10px;\n  }\n}",
  "markup": "<div class=\"wrap\">\n  <h1>Crisis Dashboard: CEOs</h1>\n\n  <div class=\"controls-row\">\n    <div class=\"refresh-panel\">\n      <button id=\"refreshBtn\">Refresh Data</button>\n    </div>\n  </div>\n\n  <div class=\"controls card\">\n    <div class=\"controls-grid controls-header\">\n      <div class=\"muted field-label\">Date</div>\n      <div class=\"muted field-label\">Filter (CEO or Company)</div>\n      <div class=\"muted field-label\">Sector</div>\n      <div class=\"muted field-label\">Company Size</div>\n      <div class=\"field-label field-label-empty\" aria-hidden=\"true\"></div>\n      <div class=\"field-label field-label-empty\" aria-hidden=\"true\"></div>\n    </div>\n    <div class=\"controls-grid controls-inputs\">\n      <div class=\"date-input-stack\">\n        <select id=\"dateSelect\"></select>\n        <div class=\"lookback-controls\" role=\"group\" aria-label=\"Chart lookback window\">\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"30\" title=\"Show last 30 days\">30d</button>\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"45\" title=\"Show last 45 days\">45d</button>\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"60\" title=\"Show last 60 days\">60d</button>\n          <button type=\"button\" class=\"lookback-btn\" data-days=\"90\" title=\"Show last 90 days\">90d</button>\n        </div>\n      </div>\n      <input id=\"filterInput\" type=\"text\" placeholder=\"Type to filter\u2026\" />\n      <select id=\"sectorFilterSelect\" title=\"Filter by sector or industry\">\n        <option value=\"\">All sectors</option>\n      </select>\n      <select id=\"companySizeSelect\">\n        <option value=\"all\">All companies</option>\n        <option value=\"favorites\">Favorites</option>\n        <option value=\"fortune500\">Fortune 500</option>\n        <option value=\"fortune1000\">Fortune 1000</option>\n        <option value=\"forbes\">Forbes</option>\n      </select>\n      <button id=\"clearBtn\">Clear Selection</button>\n      <button id=\"crisisBtn\" class=\"filter-toggle crisis-toggle\" title=\"CEOs with more than 4 negative Top stories URLs on any single day in the last 30 days\">Crisis</button>\n    </div>\n    <div class=\"load-status\" id=\"loadStatus\">\n      <div class=\"load-bar\"><div class=\"load-bar-fill\" id=\"loadBarFill\"></div></div>\n      <div class=\"load-items\" id=\"loadItems\"></div>\n    </div>\n  </div>\n\n  <div class=\"grid\">\n    <div class=\"chart-card loading\" data-chart=\"composite\">\n      <h3>Negative Signal Composite (SERP Features vs News)</h3>\n      <div class=\"chart-actions\">\n        <span class=\"dates-range muted\" aria-live=\"polite\"></span>\n        <div class=\"dates-pager\">\n          <button class=\"dates-prev\" title=\"Older window\">\u2190</button>\n          <button class=\"dates-next\" title=\"Newer window\">\u2192</button>\n        </div>\n      </div>\n      <canvas id=\"negativeCompositeChart\"></canvas>\n    </div>\n\n    <div class=\"chart-card loading\" data-chart=\"features\">\n      <h3>SERP Feature Negativity (stacked)</h3>\n      <div class=\"chart-actions\">\n        <span class=\"dates-range muted\" aria-live=\"polite\"></span>\n        <div class=\"dates-pager\">\n          <button class=\"dates-prev\" title=\"Older window\">\u2190</button>\n          <button class=\"dates-next\" title=\"Newer window\">\u2192</button>\n        </div>\n      </div>\n      <canvas id=\"featureChart\"></canvas>\n    </div>\n\n    <div class=\"chart-card loading\" data-chart=\"features\">\n      <h3>SERP Feature Control (stacked)</h3>\n      <div class=\"chart-actions\">\n        <span class=\"dates-range muted\" aria-live=\"polite\"></span>\n        <div class=\"dates-pager\">\n          <button class=\"dates-prev\" title=\"Older window\">\u2190</button>\n          <button class=\"dates-next\" title=\"Newer window\">\u2192</button>\n        </div>\n      </div>\n      <canvas id=\"featureControlChart\"></canvas>\n    </div>\n    <div class=\"chart-card loading\" data-chart=\"news\">\n      <h3>News sentiment (percent of articles)</h3>\n      <div class=\"chart-actions\">\n        <span class=\"dates-range muted\" aria-live=\"polite\"></span>\n        <div class=\"dates-pager\">\n          <button class=\"dates-prev\" title=\"Older window\">\u2190</button>\n          <button class=\"dates-next\" title=\"Newer window\">\u2192</button>\n        </div>\n      </div>\n      <canvas id=\"newsChart\"></canvas>\n    </div>\n    <div class=\"chart-card loading\" data-chart=\"serps\">\n      <h3>Organic Search (Negative % \u2022 Control %)</h3>\n      <div class=\"chart-actions\">\n        <span class=\"dates-range muted\" aria-live=\"polite\"></span>\n        <div class=\"dates-pager\">\n          <button class=\"dates-prev\" title=\"Older window\">\u2190</button>\n          <button class=\"dates-next\" title=\"Newer window\">\u2192</button>\n        </div>\n      </div>\n      <canvas id=\"serpChart\"></canvas>\n    </div>\n  </div>\n\n  <div class=\"table-card\">\n    <p class=\"table-hint\">Select a CEO using the checkbox to view their trend lines. Click the sparkline to see the company's 30-day stock chart with negative article markers.</p>\n    <div class=\"table-scroll\">\n      <table>\n        <thead>\n          <tr>\n            <th>CEO</th>\n            <th data-key=\"stock_price\">\n              <div class=\"tooltip-header\">\n                Stock Price\n                <span class=\"tooltip-text\">Current stock price from today's market open (or latest available)</span>\n              </div>\n            </th>\n            <th data-key=\"daily_change\">\n              <div class=\"tooltip-header\">\n                Overnight<br>Change % \u25b2\u25bc\n                <span class=\"tooltip-text\">Percentage change from yesterday's close to today's market open (overnight gap)</span>\n              </div>\n            </th>            \n            <th data-key=\"trend\">\n              <div class=\"tooltip-header\">\n                30-Day Trend\n                <span class=\"tooltip-text\">Visual sparkline of stock price over 30 days. Click to view detailed chart with negative news articles overlaid as markers</span>\n              </div>\n            </th>\n            <th data-key=\"neg_news\">\n              <div class=\"tooltip-header\">\n                Negative<br>News % \u25b2\u25bc\n                <span class=\"tooltip-text\">Percentage of articles mentioning this CEO that carry negative sentiment. Tracks CEO-specific reputation risk</span>\n              </div>\n            </th>\n            <th data-key=\"neg_top_stories\">\n              <div class=\"tooltip-header\">\n                Negative<br>Top Stories % \u25b2\u25bc\n                <span class=\"tooltip-text\">Percentage of Top Stories results with negative sentiment for this CEO on the selected day</span>\n              </div>\n            </th>\n            <th data-key=\"neg_serp\">\n              <div class=\"tooltip-header\">\n                Negative<br>Organic % \u25b2\u25bc\n                <span class=\"tooltip-text\">Percentage of Page 1 Google search results for this CEO's name with negative sentiment. Important for executive reputation</span>\n              </div>\n            </th>\n            <th data-key=\"ctrl_pct\">\n              <div class=\"tooltip-header\">\n                SERP<br>Control % \u25b2\u25bc\n                <span class=\"tooltip-text\">Percentage of first-page results the CEO or their company can control (LinkedIn, official bio, company press). Higher = better narrative control</span>\n              </div>\n            </th>\n            <th data-key=\"risk\">\n              <div class=\"tooltip-header\">\n                Risk<br>\u25b2\u25bc\n                <span class=\"tooltip-text\">Composite risk score based on Negative Organic % and Control %. High risk = significant negative search visibility with limited control. Low risk = minimal negative presence or strong narrative control.</span>\n              </div>\n            </th>\n            <th class=\"action-col\">Coverage</th>\n            <th class=\"action-col\">Boards</th>\n          </tr>\n        </thead>\n        <tbody id=\"tbody\"></tbody>\n      </table>\n    </div>\n\n    <div class=\"pagination\">\n      <button id=\"prevBtn\">Prev</button>\n      <div>Page <span id=\"pageNo\">1</span> / <span id=\"pageTotal\">1</span></div>\n      <button id=\"nextBtn\">Next</button>\n    </div>\n  </div>\n</div>\n\n<div id=\"modal\" class=\"modal\" role=\"dialog\" aria-modal=\"true\" aria-labelledby=\"modalTitle\">\n  <div class=\"box\">\n    <header>\n      <h3 id=\"modalTitle\">Title</h3>\n      <button id=\"modalClose\">Close</button>\n    </header>\n    <div class=\"content\" id=\"modalContent\"></div>\n  </div>\n</div>\n\n<!-- Stock Chart Modal with Heatmap Legend -->\n<div id=\"stockChartModal\" class=\"modal\" role=\"dialog\" aria-modal=\"true\">\n  <div class=\"box\">\n    <header>\n      <h3 id=\"stockChartTitle\">Stock Price History</h3>\n      <button id=\"volumeToggle\" class=\"volume-toggle\">\ud83d\udcca Volume</button>\n      <button id=\"trendsToggle\" class=\"trends-toggle\">\ud83d\udd0d Trends</button>\n      <span id=\"stockChartClose\" class=\"close\">&times;</span>\n    </header>\n    <div class=\"content\">\n      <canvas id=\"stockChart\"></canvas>\n      <div id=\"heatmapLegend\" class=\"heatmap-legend\" style=\"display: none;\">\n        <div class=\"heatmap-legend-item\">\n          <div class=\"heatmap-legend-dot medium\"></div>\n          <span>3-5 Negative Articles</span>\n        </div>\n        <div class=\"heatmap-legend-item\">\n          <div class=\"heatmap-legend-dot high\"></div>\n          <span>6+ Negative Articles</span>\n        </div>\n      </div>\n    </div>\n  </div>\n</div>",
  "externalScripts": [
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
    "https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js",
    "https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js",
    "/static/app/shared-data.js",
    "/static/app/legacy-shared.js"
  ]
};

function runInlineScript(scope) {
  const {
    window,
    self,
    top,
    parent,
    document,
    history,
    location,
    globalThis,
    fetch,
    localStorage,
    sessionStorage,
    navigator,
    requestAnimationFrame,
    cancelAnimationFrame,
    requestIdleCallback,
    cancelIdleCallback,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    performance,
    console,
    URL,
    URLSearchParams,
    Chart,
    Papa,
    ResizeObserver,
    MutationObserver,
    Event,
    CustomEvent,
    Node,
    HTMLElement,
    HTMLCanvasElement,
    CSS,
    getComputedStyle
  } = scope;

  // CLOUD STORAGE CONFIGURATION - ADD THIS AT TOP
  // DATA CONFIGURATION - Uses Cloud Run proxy for authenticated access
  // The proxy endpoints in app.py fetch from private GCS bucket
  function getDataUrl(path) {
    // Route to appropriate proxy endpoint based on path
    if (path.startsWith('data/')) {
      // /api/data/daily_counts/..., /api/data/processed_articles/..., etc.
      return '/api/data/' + path.substring(5);
    } else if (path.startsWith('rosters/')) {
      // /api/data/rosters/main-roster.csv
      return '/api/data/' + path;
    }
    return '/api/data/' + path;
  }
  // END OF CLOUD STORAGE CONFIG

  const DEFAULT_DAYS = 30;
  const EXTENDED_DAYS = 90;
  const LOOKBACK_OPTIONS = [30, 45, 60, 90];
  const CRISIS_DAYS = 30;
  const CRISIS_MIN_NEG = 4;
  let currentDays = DEFAULT_DAYS;
  let extendingHistoryPromise = null;
  let extendedHistoryQueued = false;
  const COUNTS_PATH = () => `/api/v1/daily_counts?kind=ceo_articles&days=${currentDays}`;
  const ROSTER_CANDIDATES   = ['/api/v1/roster'];
  const BOARDS_CSV          = ['/api/v1/boards'];
  const SERPS_DAILY_CSV     = () => `/api/v1/daily_counts?kind=ceo_serps&days=${currentDays}`;
  const SERP_FEATURES_INDEX_PATH  = () => `/api/v1/serp_features?entity=ceo&days=${currentDays}&mode=index`;
  const SERP_FEATURES_TOP_STORIES_PATH = (date) => `/api/v1/serp_features?entity=ceo&days=${currentDays}&feature_type=top_stories_items${date ? `&date=${date}` : ''}`;
  const SERP_FEATURES_CRISIS_KEYS_PATH = () => `/api/v1/serp_features?entity=ceo&days=${CRISIS_DAYS}&feature_type=top_stories_items`;
  const SERP_FEATURES_ENTITY_PATH = name => `/api/v1/serp_features?entity=ceo&days=${currentDays}&entity_name=${encodeURIComponent(name)}`;
  const SERP_FEATURES_CONTROL_INDEX_PATH  = () => `/api/v1/serp_feature_controls?entity=ceo&days=${currentDays}&mode=index`;
  const SERP_FEATURES_CONTROL_ENTITY_PATH = name => `/api/v1/serp_feature_controls?entity=ceo&days=${currentDays}&entity_name=${encodeURIComponent(name)}`;
  const NEGATIVE_SUMMARY_PATH = '/api/v1/negative_summary?days=30';
  const PRELOAD_MODAL_DATA = false;
  const MODAL_LIMIT = 200;
  const ARTICLES_DAILY_PATH = d => `/api/v1/processed_articles?date=${d}&entity=ceo&kind=table`;
  const HEADLINES_PATH_CEO  = (d, ceo, offset = 0) =>
    `/api/v1/processed_articles?date=${d}&entity=ceo&kind=modal&entity_name=${encodeURIComponent(ceo)}&limit=${MODAL_LIMIT}&offset=${offset}`;
  const SERP_PROCESSED_PATH = d => `/api/v1/processed_serps?date=${d}&entity=ceo&kind=table`;
  const SERP_ROWS_PATH      = (d, ceo, offset = 0) =>
    `/api/v1/processed_serps?date=${d}&entity=ceo&kind=modal&entity_name=${encodeURIComponent(ceo)}&limit=${MODAL_LIMIT}&offset=${offset}`;
  const FEATURE_ALL_KEY = 'all';
  const FEATURE_MODAL_ORDER = ['organic','aio_citations','paa_items','videos_items','perspectives_items','top_stories_items'];
  const FEATURE_MODAL_LIMIT = 100;
  const SERP_FEATURE_ITEMS_PATH = (d, ceo, feature, offset = 0, limit = FEATURE_MODAL_LIMIT) => {
    const featParam = feature ? `&feature_type=${encodeURIComponent(feature)}` : '';
    return `/api/v1/serp_feature_items?date=${d}&entity=ceo&entity_name=${encodeURIComponent(ceo)}${featParam}&limit=${limit}&offset=${offset}`;
  };

  let rosterMap = new Map();
  let companyToCeo = new Map();
  let allCountsRows = [];
  let serpsDaily = [];
  let serpFeatureIndexRows = [];
  let serpFeatureEntityRows = [];
  let serpFeatureControlIndexRows = [];
  let serpFeatureControlEntityRows = [];
  let topStoriesNegByDate = new Map();
  let crisisEntityKeys = new Set();
  let boardsByCeo = new Map();
  let filteredRows = [];
  let selectedCeo = null;
  let currentSort = { key:null, dir:1 };
  let currentPage = 1;
  let dataReady = false;
  const PAGE_SIZE = 25;
  const CHART_ORDER_STORAGE_KEY = `riskdash.chart_order:${window.location.pathname}`;
  const CHART_COLLAPSE_STORAGE_KEY = `riskdash.chart_collapse:${window.location.pathname}`;
  const COMPOSITE_CHART_KEY = 'negativeCompositeChart';
  let draggedChartCard = null;

  function getChartGrid(){
    return document.querySelector('.grid');
  }
  function getChartCards(){
    const grid = getChartGrid();
    return grid ? Array.from(grid.querySelectorAll('.chart-card[data-chart]')) : [];
  }
  function getChartCardKey(card){
    if (!card) return '';
    if (!card.dataset.chartKey){
      const canvasId = String(card.querySelector('canvas[id]')?.id || '').trim();
      card.dataset.chartKey = canvasId || `chart-${Math.random().toString(36).slice(2,8)}`;
    }
    return card.dataset.chartKey;
  }
  function loadChartCollapsePreference(){
    try {
      const raw = localStorage.getItem(CHART_COLLAPSE_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      if (parsed && typeof parsed === 'object') return parsed;
    } catch {}
    return {};
  }
  function saveChartCollapsePreference(pref){
    try { localStorage.setItem(CHART_COLLAPSE_STORAGE_KEY, JSON.stringify(pref)); } catch {}
  }
  function setChartCollapsed(card, collapsed){
    const key = getChartCardKey(card);
    if (!key) return;
    const pref = loadChartCollapsePreference();
    pref[key] = !!collapsed;
    saveChartCollapsePreference(pref);
    card.classList.toggle('chart-collapsed', !!collapsed);
  }
  function applyChartCollapsePreference(){
    const pref = loadChartCollapsePreference();
    getChartCards().forEach(card => {
      const key = getChartCardKey(card);
      card.classList.toggle('chart-collapsed', !!pref[key]);
    });
  }
  function bindChartCollapse(){
    getChartCards().forEach(card => {
      if (card.dataset.collapseBound === '1') return;
      card.dataset.collapseBound = '1';
      const titleEl = card.querySelector('h3');
      if (!titleEl) return;
      titleEl.classList.add('chart-collapsible-title');
      titleEl.title = 'Click to collapse or expand';
      titleEl.addEventListener('click', (event) => {
        const target = (event.target && typeof event.target.closest === 'function') ? event.target : null;
        if (target && target.closest('.drag-handle')) return;
        event.preventDefault();
        const willCollapse = !card.classList.contains('chart-collapsed');
        setChartCollapsed(card, willCollapse);
        if (!willCollapse && dataReady && typeof renderCharts === 'function') {
          requestAnimationFrame(() => { renderCharts(); });
        }
      });
    });
  }
  function saveChartOrderPreference(){
    const keys = getChartCards().map(getChartCardKey).filter(Boolean);
    if (!keys.length) return;
    try { localStorage.setItem(CHART_ORDER_STORAGE_KEY, JSON.stringify(keys)); } catch {}
  }
  function applyChartOrderPreference(){
    const grid = getChartGrid();
    if (!grid) return;
    const cards = getChartCards();
    const byKey = new Map(cards.map(card => [getChartCardKey(card), card]));
    let saved = [];
    try {
      const raw = localStorage.getItem(CHART_ORDER_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed)) saved = parsed.map(v => String(v || '')).filter(Boolean);
    } catch {}
    if (byKey.has(COMPOSITE_CHART_KEY) && !saved.includes(COMPOSITE_CHART_KEY)) {
      saved = [COMPOSITE_CHART_KEY, ...saved];
    }
    saved = [...new Set(saved)];
    if (!saved.length) return;
    const ordered = [];
    saved.forEach(key => {
      const card = byKey.get(key);
      if (card) {
        ordered.push(card);
        byKey.delete(key);
      }
    });
    byKey.forEach(card => ordered.push(card));
    ordered.forEach(card => grid.appendChild(card));
  }
  function bindChartReorder(){
    const cards = getChartCards();
    cards.forEach(card => {
      getChartCardKey(card);
      if (card.dataset.reorderBound === '1') return;
      card.dataset.reorderBound = '1';
      card.setAttribute('draggable', 'true');
      card.removeAttribute('title');
      card.setAttribute('aria-label', 'Drag to reorder chart card');
      const titleEl = card.querySelector('h3');
      if (titleEl && !titleEl.querySelector('.drag-handle')){
        const handle = document.createElement('span');
        handle.className = 'drag-handle';
        handle.title = 'Drag to reorder charts';
        handle.setAttribute('aria-hidden', 'true');
        titleEl.insertBefore(handle, titleEl.firstChild);
      }
      card.addEventListener('dragstart', () => {
        draggedChartCard = card;
        card.classList.add('dragging');
      });
      card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
        draggedChartCard = null;
        getChartCards().forEach(el => el.classList.remove('drag-over'));
        saveChartOrderPreference();
      });
      card.addEventListener('dragover', (event) => {
        event.preventDefault();
        if (!draggedChartCard || draggedChartCard === card) return;
        const rect = card.getBoundingClientRect();
        const before = event.clientY < (rect.top + rect.height / 2);
        const parent = card.parentNode;
        if (!parent) return;
        card.classList.add('drag-over');
        parent.insertBefore(draggedChartCard, before ? card : card.nextSibling);
      });
      card.addEventListener('dragleave', () => {
        card.classList.remove('drag-over');
      });
      card.addEventListener('drop', (event) => {
        event.preventDefault();
        card.classList.remove('drag-over');
      });
    });
  }
  function initChartOrderPreference(){
    const grid = getChartGrid();
    if (!grid) return;
    if (grid.dataset.chartOrderInit === '1') return;
    applyChartOrderPreference();
    bindChartReorder();
    bindChartCollapse();
    applyChartCollapsePreference();
    grid.dataset.chartOrderInit = '1';
  }

  function updateChartSkeletons(){
    const compositeState = (_loadState.news === 'done' && _loadState.features === 'done')
      ? 'done'
      : ((_loadState.news === 'error' || _loadState.features === 'error') ? 'error' : 'loading');
    const map = {
      news: _loadState.news,
      serps: _loadState.serps,
      features: _loadState.features,
      composite: compositeState
    };
    document.querySelectorAll('.chart-card[data-chart]').forEach(card=>{
      const key = card.getAttribute('data-chart');
      const state = map[key];
      card.classList.toggle('loading', state !== 'done');
    });
  }

  function updateDateOptions(){
    const dateSet = new Set([ ...allCountsRows.map(r=>r.date), ...serpsDaily.map(r=>r.date) ].filter(isISODate));
    const dates = [...dateSet].sort();
    const sel = document.getElementById('dateSelect');
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = dates.map(d=>`<option value="${d}">${d}</option>`).join('');
    sel.value = dates.includes(prev) ? prev : (dates[dates.length-1] || '');
    updateLookbackButtons();
  }

  function updateLookbackButtons(){
    document.querySelectorAll('.lookback-btn').forEach(btn => {
      const days = Number(btn.dataset.days || DEFAULT_DAYS);
      const active = days === currentDays;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  function bindLookbackControls(){
    document.querySelectorAll('.lookback-btn').forEach(btn => {
      if (btn.dataset.boundLookback === '1') return;
      btn.dataset.boundLookback = '1';
      btn.addEventListener('click', async () => {
        await setLookbackDays(btn.dataset.days);
      });
    });
    updateLookbackButtons();
  }

  function setLookbackLoadState(){
    ['news', 'serps', 'features'].forEach(key => {
      if (Object.prototype.hasOwnProperty.call(_loadState, key)) {
        setLoadStatus(key, 'loading');
      }
    });
  }

  async function reloadLookbackData(){
    const tasks = [loadCounts(), loadSerpDaily(), loadSerpFeaturesIndex()];
    if (crisisOnly || crisisEntityKeys.size) tasks.push(loadCrisisEntityKeys());
    await Promise.all(tasks);
    const selectedDate = String(document.getElementById('dateSelect')?.value || '').trim();
    if (selectedDate) await loadTopStoriesForDate(selectedDate);
    if (selectedCeo) await loadSerpFeaturesForEntity(selectedCeo);
    await renderAll();
  }

  function normalizeLookbackDays(days){
    const next = Number(days);
    return LOOKBACK_OPTIONS.includes(next) ? next : DEFAULT_DAYS;
  }

  async function setLookbackDays(days, { force = false } = {}){
    const nextDays = normalizeLookbackDays(days);
    if (!force && nextDays === currentDays) {
      updateLookbackButtons();
      return false;
    }
    currentDays = nextDays;
    DATE_WINDOW_SIZE = nextDays;
    dateWindowStart = null;
    dateWindowPinned = false;
    extendingHistoryPromise = null;
    extendedHistoryQueued = currentDays >= EXTENDED_DAYS;
    _serpFeatureCache.clear();
    _serpFeatureControlCache.clear();
    topStoriesNegByDate = new Map();
    updateLookbackButtons();
    setLookbackLoadState();
    await reloadLookbackData();
    return true;
  }

  function queueExtendedHistoryLoad(){ return; }

  async function ensureExtendedHistory(){
    return setLookbackDays(EXTENDED_DAYS, { force: currentDays !== EXTENDED_DAYS });
  }


  function maybeRenderAll(){
    const sel = document.getElementById('dateSelect');
    if (_loadState.news !== 'done' || _loadState.serps !== 'done') return;
    if (!sel.value) return;
    if (!dataReady) {
      dataReady = true;
    }
    updateChartSkeletons();
    renderAll();
  }

  function chartsDataReady(){
    return _loadState.news !== 'loading' &&
      _loadState.serps !== 'loading' &&
      _loadState.features !== 'loading';
  }

  function updateEntityUrlParam(param, value){
    try{
      const url = new URL(window.location.href);
      if (value) url.searchParams.set(param, value);
      else url.searchParams.delete(param);
      url.searchParams.delete('company');
      const next = `${url.pathname}${url.search}${url.hash}`;
      const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
      if (current !== next) {
        window.history.replaceState({}, '', next);
        const after = `${window.location.pathname}${window.location.search}${window.location.hash}`;
        if (after !== next) {
          window.history.pushState({}, '', next);
        }
      }
      if (window.top && window.top !== window) {
        const topUrl = new URL(window.top.location.href);
        topUrl.searchParams.set('tab', 'ceos');
        if (value) topUrl.searchParams.set(param, value);
        else topUrl.searchParams.delete(param);
        topUrl.searchParams.delete('company');
        const topNext = `${topUrl.pathname}${topUrl.search}${topUrl.hash}`;
        window.top.history.replaceState({}, '', topNext);
      }
      console.log('🔗 URL updated', param, value, next);
    }catch{}
  }

  async function selectCeo(name){
    if (!name) return;
    selectedCeo = name;
    updateEntityUrlParam('ceo', selectedCeo);
    await loadSerpFeaturesForEntity(selectedCeo);
    renderTable();
    if (chartsDataReady()) renderCharts();
  }

  let newsChart, serpChart, stockChart, featureChart, featureControlChart, negativeCompositeChart;
  let globalStockData = {};
  let showVolume = false; // Toggle state for volume display
  let showTrends = false; // Toggle state for trends display
  let globalTrendsData = {}; // Trends data by company
  let companyTickers = new Map();
  let favoriteCeos = new Set();
  let favoriteCompanies = new Set();
  let fortuneFlags = new Map();
  let companySector = new Map();
  let companySizeFilter = 'all';
  let crisisOnly = false;

  const _articlesCache = new Map();
  const _serpAggCache  = new Map();
  const _serpFeatureCache = new Map();
  const _serpFeatureControlCache = new Map();
  const FEATURE_ORDER_SENTIMENT = ['organic','aio_citations','paa_items','videos_items','perspectives_items','top_stories_items'];
  const FEATURE_ORDER_BASE = ['organic','aio','paa','videos','perspectives','top_stories'];
  const FEATURE_ITEM_MAP = {
    aio: 'aio_citations',
    paa: 'paa_items',
    videos: 'videos_items',
    perspectives: 'perspectives_items',
    top_stories: 'top_stories_items'
  };
  const FEATURE_LABELS = {
    all: 'All Features',
    organic: 'Organic',
    aio_citations: 'AIO citations',
    paa_items: 'PAA',
    videos_items: 'Videos',
    perspectives_items: 'Perspectives',
    aio: 'AI Overview',
    paa: 'People also ask',
    videos: 'Videos',
    perspectives: 'Perspectives',
    top_stories_items: 'Top stories',
    top_stories: 'Top stories'
  };
  const FEATURE_COLORS = ['#ff3b30', '#ff5e57', '#ff7a59', '#ff9f43', '#ff6f91', '#e63946'];
  const CONTROL_COLORS = ['#2d9cdb', '#1b84d1', '#1769aa', '#115293', '#0d3b66', '#0a2742'];
  const mapFeatureType = (feature) => FEATURE_ITEM_MAP[feature] || feature;
  const isItemFeature = (feature) => {
    const mapped = mapFeatureType(feature);
    return mapped === feature && FEATURE_ORDER_SENTIMENT.includes(mapped);
  };
  function refreshCrisisEntityKeys(rows = []){
    const sourceRows = Array.isArray(rows) ? rows : [];
    const negByEntityDay = new Map();
    const crisisKeys = new Set();
    sourceRows.forEach(r=>{
      const feature = mapFeatureType(r.feature || r.feature_type);
      if (feature !== 'top_stories_items') return;
      const date = String(r.date || '').trim();
      if (!isISODate(date)) return;
      const total = +r.total || +r.total_count || 0;
      const neg = +r.neg || +r.negative_count || 0;
      if (total <= 0 || neg <= 0) return;
      const entityKey = normEntityKey(r.entity || r.entity_name);
      if (!entityKey) return;
      const key = `${entityKey}|${date}`;
      const dailyNeg = (negByEntityDay.get(key) || 0) + neg;
      negByEntityDay.set(key, dailyNeg);
      if (dailyNeg > CRISIS_MIN_NEG) crisisKeys.add(entityKey);
    });
    crisisEntityKeys = crisisKeys;
  }

  async function loadCrisisEntityKeys(){
    try{
      const rows = await fetchCsv(SERP_FEATURES_CRISIS_KEYS_PATH());
      refreshCrisisEntityKeys(rows);
    }catch(e){
      crisisEntityKeys = new Set();
    }
  }
  function matchesCompanySize(company, ceo){
    const flags = fortuneFlags.get(company) || {};
    if (companySizeFilter === 'favorites') return favoriteCeos.has(ceo);
    if (companySizeFilter === 'fortune500') return !!flags.f500;
    if (companySizeFilter === 'fortune1000') return !!flags.f1000;
    if (companySizeFilter === 'forbes') return !!flags.forbes;
    return true;
  }
  const _loadSections = ['roster','news','serps','features','boards','stock','trends','negative'];
  const _loadLabels = {
    roster: 'Roster',
    news: 'News sentiment',
    serps: 'SERP metrics',
    features: 'SERP features',
    boards: 'Boards data',
    stock: 'Stock data',
    trends: 'Trends data',
    negative: 'Negative summary'
  };
  let _loadState = {};
  let _dataRendered = false;

  function initLoadStatus(){
    _loadState = Object.fromEntries(_loadSections.map(k => [k, 'loading']));
    const items = document.getElementById('loadItems');
    const wrap = document.getElementById('loadStatus');
    if (wrap) wrap.style.display = 'block';
    if (items){
      items.innerHTML = _loadSections.map(k => `<span class="load-pill" data-key="${k}">${_loadLabels[k] || k}: loading</span>`).join('');
    }
    updateLoadBar();
  }

  function setLoadStatus(key, state){
    _loadState[key] = state;
    const pill = document.querySelector(`.load-pill[data-key="${key}"]`);
    if (pill){
      pill.textContent = `${_loadLabels[key] || key}: ${state}`;
      pill.classList.toggle('done', state === 'done');
      pill.classList.toggle('error', state === 'error');
    }
    updateLoadBar();
    updateChartSkeletons();
  }

  function updateLoadBar(){
    const total = _loadSections.length;
    const done = Object.values(_loadState).filter(v => v === 'done').length;
    const fill = document.getElementById('loadBarFill');
    if (fill){
      fill.style.width = `${Math.round((done / total) * 100)}%`;
    }
    maybeHideLoadStatus();
  }

  function maybeHideLoadStatus(){
    const total = _loadSections.length;
    const done = Object.values(_loadState).filter(v => v === 'done').length;
    const wrap = document.getElementById('loadStatus');
    if (wrap && done === total && _dataRendered){
      wrap.style.display = 'none';
    }
  }

  let DATE_WINDOW_SIZE = currentDays;
  let dateWindowStart = null;
  let dateWindowPinned = false;
  const negativeSummaryCache = new Map();
  let negativeSummaryIndex = null;
  const NEGATIVE_ARTICLE_THRESHOLD = 3;

  const isISODate = s => /^\d{4}-\d{2}-\d{2}$/.test(String(s||'').trim());
  function articlePublishedMeta(row){
    const raw = String(row?.published_date || row?.published_at || '').trim();
    if (!raw) return '';
    const match = raw.match(/^(\d{4}-\d{2}-\d{2})(?:[T ].*)?$/);
    if (match) return `Published ${match[1]}`;
    if (/[tT]/.test(raw)) {
      const parsed = new Date(raw);
      if (!Number.isNaN(parsed.getTime())) return `Published ${parsed.toISOString().slice(0, 10)}`;
    }
    return `Published ${raw}`;
  }
  function buildSerpMetaLabel(row, fallbackSource = ''){
    const source = String(row?.source || fallbackSource || '').trim();
    const published = articlePublishedMeta(row);
    return [source, published].filter(Boolean).join(' · ');
  }
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m]));
  function fmtPct(v){ 
    return (v==null) ? 'N/A' : Math.round(v*100) + '%'; 
  }

  const canonName = s => String(s || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();

  async function loadStockData() {
    try {
      console.log('📊 Looking for most recent stock data...');

      const latest = await window.CrisisDashboardData.fetchLatestDatedCsv({
        key: 'stock_data',
        buildUrl: (dateStr) => `/api/v1/stock_data?date=${dateStr}`,
        maxDays: 7,
        transform: (rows) => parseStockRows(rows),
      });

      if (latest) {
        console.log(`✅ Successfully loaded stock data from ${latest.date} (${latest.daysBack} day(s) old)`);
        console.log(`📈 Found data for ${Object.keys(latest.data).length} companies`);
        return latest.data;
      }
    } catch (error) {
      console.error('❌ STOCK DATA LOAD FAILED:', error);
      setLoadStatus('stock', 'error');
      return {};
    }
  }

  function parseStockRows(rows) {
    const stockData = {};
    const toFloat = (val) => {
      if (val === null || val === undefined || val === '') return null;
      const num = parseFloat(val);
      return Number.isFinite(num) ? num : null;
    };
    rows.forEach(r => {
      const company = String(r.company || '').trim();
      if (!company) return;
      const priceHistory = Array.isArray(r.price_history)
        ? r.price_history
        : String(r.price_history || '').split('|').filter(Boolean).map(Number).filter(n => Number.isFinite(n));
      const dateHistory = Array.isArray(r.date_history)
        ? r.date_history
        : String(r.date_history || '').split('|').filter(Boolean);
      const volumeHistory = Array.isArray(r.volume_history)
        ? r.volume_history
        : String(r.volume_history || '').split('|').filter(Boolean).map(Number).filter(n => Number.isFinite(n));
      stockData[company] = {
        ticker: String(r.ticker || ''),
        company,
        openingPrice: toFloat(r.opening_price ?? r.openingPrice),
        dailyChange: toFloat(r.daily_change_pct ?? r.dailyChange ?? r.daily_change),
        sevenDayChange: toFloat(r.seven_day_change_pct ?? r.sevenDayChange),
        priceHistory,
        dateHistory,
        volumeHistory,
        lastUpdated: r.last_updated || ''
      };
    });
    return stockData;
  }

  function createSparkline(prices, isPositive) {
    if (!prices || prices.length < 2) return null;
    
    const canvas = document.createElement('canvas');
    canvas.width = 80;
    canvas.height = 24;
    canvas.className = 'sparkline';
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 2;
    
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    
    const stepX = (width - padding * 2) / (prices.length - 1);
    
    ctx.beginPath();
    ctx.strokeStyle = isPositive ? '#82c616' : '#ff8261';
    ctx.lineWidth = 1.5;
    
    prices.forEach((price, i) => {
      const x = padding + i * stepX;
      const y = height - padding - ((price - min) / range) * (height - padding * 2);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    return canvas;
  }

  async function loadNegativeSummaryIndex() {
    const logLabel = 'negative-summary';
    console.time(logLabel);
    console.log('📰 Loading negative articles summary...');
    setLoadStatus('negative', 'loading');

    try {
      const rows = await fetchCsv(`${NEGATIVE_SUMMARY_PATH}&mode=index`);
      console.log(`📊 Negative summary rows: ${rows.length}`);
      console.log(`📊 Loaded ${rows.length} total rows from summary file`);

      negativeSummaryIndex = rows;
      console.log(`✅ Loaded negative summary index rows: ${rows.length}`);
      console.timeEnd(logLabel);
      setLoadStatus('negative', 'done');
      return rows;
    } catch (e) {
      console.error('❌ Failed to load negative summary:', e);
      console.timeEnd(logLabel);
      setLoadStatus('negative', 'error');
      return [];
    }
  }

  async function loadNegativeSummaryForCompany(company) {
    if (!company) return [];
    if (negativeSummaryCache.has(company)) return negativeSummaryCache.get(company);
    try {
      const url = `${NEGATIVE_SUMMARY_PATH}&company=${encodeURIComponent(company)}`;
      const rows = await fetchCsv(url);
      const items = [];
      rows.forEach(r => {
        const count = parseInt(r.negative_count) || 0;
        if (count < NEGATIVE_ARTICLE_THRESHOLD) return;
        items.push({
          date: String(r.date || '').trim(),
          count: count,
          headlines: (r.top_headlines || '').split('|').filter(h => h.trim()),
          type: String(r.article_type || 'unknown').trim()
        });
      });
      negativeSummaryCache.set(company, items);
      return items;
    } catch (e) {
      return [];
    }
  }


  // Load Google Trends data
  async function loadTrendsData() {
    try {
      console.log('🔍 Looking for most recent trends data...');

      const latest = await window.CrisisDashboardData.fetchLatestDatedCsv({
        key: 'trends_data',
        buildUrl: (dateStr) => `/api/v1/trends_data?date=${dateStr}`,
        maxDays: 7,
        transform: (rows) => parseTrendsRows(rows),
      });

      if (latest) {
        console.log(`✅ Successfully loaded trends data from ${latest.date} (${latest.daysBack} day(s) old)`);
        console.log(`📈 Found trends for ${Object.keys(latest.data).length} companies`);
        setLoadStatus('trends', 'done');
        return latest.data;
      }
    } catch (error) {
      console.error('❌ TRENDS DATA LOAD FAILED:', error);
      setLoadStatus('trends', 'error');
      return {};
    }
  }

  function parseTrendsRows(rows) {
    const trendsData = {};
    rows.forEach(r => {
      const company = String(r.company || '').trim();
      if (!company) return;
      const trendsHistory = Array.isArray(r.trends_history)
        ? r.trends_history
        : String(r.trends_history || '').split('|').filter(Boolean).map(Number).filter(n => Number.isFinite(n));
      const dateHistory = Array.isArray(r.date_history)
        ? r.date_history
        : String(r.date_history || '').split('|').filter(Boolean);
      trendsData[company] = {
        company,
        trendsHistory,
        dateHistory,
        lastUpdated: r.last_updated || ''
      };
    });
    return trendsData;
  }

  async function showStockChart(company) {
    const stock = globalStockData[company];
    if (!stock || !stock.priceHistory || !stock.priceHistory.length) {
      alert('No stock data available for this company');
      return;
    }
    
    const modal = document.getElementById('stockChartModal');
    document.getElementById('stockChartTitle').textContent = 
      `${company} (${stock.ticker}) - 30-Day Price History`;
    modal.classList.add('open');
    
    if (stockChart) stockChart.destroy();
    
    const negativeDates = await loadNegativeSummaryForCompany(company);
    const relevantNegatives = negativeDates.filter(item => 
      stock.dateHistory.includes(item.date)
    );
    
    const negativesByDate = new Map();
    relevantNegatives.forEach(item => {
      if (!negativesByDate.has(item.date)) {
        negativesByDate.set(item.date, {
          totalCount: 0,
          ceoCount: 0,
          brandCount: 0,
          allHeadlines: []
        });
      }
      const agg = negativesByDate.get(item.date);
      agg.totalCount += item.count;
      if (item.type === 'ceo') agg.ceoCount += item.count;
      if (item.type === 'brand') agg.brandCount += item.count;
      agg.allHeadlines.push(...item.headlines);
    });
    
    console.log(`📊 ${company}: ${negativesByDate.size} dates with ${NEGATIVE_ARTICLE_THRESHOLD}+ negative articles`);
    
    const legendEl = document.getElementById('heatmapLegend');
    if (negativesByDate.size > 0) {
      legendEl.style.display = 'flex';
    } else {
      legendEl.style.display = 'none';
    }
    
    const annotations = {};
    Array.from(negativesByDate.entries()).forEach(([date, data], idx) => {
      const priceIndex = stock.dateHistory.indexOf(date);
      if (priceIndex === -1) return;
      
      const price = stock.priceHistory[priceIndex];
      const intensity = data.totalCount >= 6 ? 'high' : 'medium';
      
      annotations[`negArticle${idx}`] = {
        type: 'point',
        xValue: date,
        yValue: price,
        yScaleID: 'yPrice',
        backgroundColor: intensity === 'high' ? '#ff4444' : '#ffaa44',
        borderColor: '#ffffff',
        borderWidth: 2,
        radius: Math.min(data.totalCount + 3, 14),
        enter(context) {
          context.element.options.radius = Math.min(data.totalCount + 5, 18);
          context.element.options.borderWidth = 3;
          stockChart.update('none');
          return true;
        },
        leave(context) {
          context.element.options.radius = Math.min(data.totalCount + 3, 14);
          context.element.options.borderWidth = 2;
          stockChart.update('none');
          return true;
        }
      };
    });
    
    const ctx = document.getElementById('stockChart').getContext('2d');
    // Match sparkline logic: compare first to last price in 30-day window
    const firstPrice = stock.priceHistory[0];
    const lastPrice = stock.priceHistory[stock.priceHistory.length - 1];
    const isPositive = lastPrice >= firstPrice;
    
    const datasets = [{
      label: 'Closing Price ($)',
      data: stock.priceHistory,
      borderColor: isPositive ? '#82c616' : '#ff8261',
      backgroundColor: isPositive ? 'rgba(130, 198, 22, 0.1)' : 'rgba(255, 130, 97, 0.1)',
      borderWidth: 2,
      tension: 0.1,
      fill: true,
      pointRadius: 2,
      pointHoverRadius: 5,
      yAxisID: 'yPrice',
      type: 'line',
      order: 1
    }];
    
    if (showVolume && stock.volumeHistory && stock.volumeHistory.length > 0) {
      datasets.push({
        label: 'Volume',
        data: stock.volumeHistory,
        backgroundColor: 'rgba(88, 219, 237, 0.3)',
        borderColor: 'rgba(88, 219, 237, 0.5)',
        borderWidth: 1,
        yAxisID: 'yVolume',
        type: 'bar',
        order: 2,
        barPercentage: 0.9,
        categoryPercentage: 0.95
      });
    }
    
    const trendsData = globalTrendsData[company];
    if (showTrends && trendsData && trendsData.trendsHistory && trendsData.trendsHistory.length > 0) {
      datasets.push({
        label: 'Search Interest',
        data: trendsData.trendsHistory,
        borderColor: '#a855f7',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        borderWidth: 2,
        tension: 0.3,
        fill: false,
        yAxisID: 'yTrends',
        type: 'line',
        order: 0,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointBackgroundColor: '#a855f7',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 1
      });
    }
    
    stockChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: stock.dateHistory,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: { 
            display: false,
            position: 'top',
            labels: {
              color: '#ffffff',
              font: {
                size: window.innerWidth < 768 ? 11 : 13,
                weight: '500'
              },
              padding: 12,
              usePointStyle: true,
              pointStyle: 'circle',
              boxWidth: 8,
              boxHeight: 8,
              generateLabels: function(chart) {
                const labels = [
                  {
                    text: 'Stock Price',
                    fillStyle: isPositive ? '#82c616' : '#ff8261',
                    strokeStyle: isPositive ? '#82c616' : '#ff8261',
                    lineWidth: 2,
                    pointStyle: 'line'
                  }
                ];
                
                if (showVolume && stock.volumeHistory && stock.volumeHistory.length > 0) {
                  labels.push({
                    text: 'Trading Volume',
                    fillStyle: 'rgba(88, 219, 237, 0.3)',
                    strokeStyle: 'rgba(88, 219, 237, 0.5)',
                    lineWidth: 1,
                    pointStyle: 'rect'
                  });
                }
                
                if (showTrends && trendsData && trendsData.trendsHistory && trendsData.trendsHistory.length > 0) {
                  labels.push({
                    text: 'Search Interest (0-100)',
                    fillStyle: 'rgba(168, 85, 247, 0.1)',
                    strokeStyle: '#a855f7',
                    lineWidth: 2,
                    pointStyle: 'line'
                  });
                }
                
                if (negativesByDate.size > 0) {
                  let totalCeo = 0, totalBrand = 0;
                  negativesByDate.forEach(data => {
                    totalCeo += data.ceoCount;
                    totalBrand += data.brandCount;
                  });
                  
                  if (totalCeo > 0 && totalBrand > 0) {
                    labels.push({
                      text: `Negative Articles (${totalCeo} CEO, ${totalBrand} Brand)`,
                      fillStyle: '#ffaa44',
                      strokeStyle: '#ffffff',
                      lineWidth: 2,
                      pointStyle: 'circle'
                    });
                  } else if (totalCeo > 0) {
                    labels.push({
                      text: `Negative CEO Articles (${totalCeo})`,
                      fillStyle: '#ffaa44',
                      strokeStyle: '#ffffff',
                      lineWidth: 2,
                      pointStyle: 'circle'
                    });
                  } else if (totalBrand > 0) {
                    labels.push({
                      text: `Negative Brand Articles (${totalBrand})`,
                      fillStyle: '#ffaa44',
                      strokeStyle: '#ffffff',
                      lineWidth: 2,
                      pointStyle: 'circle'
                    });
                  }
                }
                
                return labels;
              }
            }
          },
          annotation: {
            annotations: annotations
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: 'rgba(255, 255, 255, 0.3)',
            borderWidth: 1,
            padding: window.innerWidth < 768 ? 10 : 14,
            displayColors: false,
            titleFont: {
              size: window.innerWidth < 768 ? 12 : 13,
              weight: 'bold'
            },
            bodyFont: {
              size: window.innerWidth < 768 ? 11 : 12
            },
            callbacks: {
              title: function(tooltipItems) {
                if (!tooltipItems || tooltipItems.length === 0) return '';
                return tooltipItems[0].label;
              },
              label: function(context) {
                const lines = [];
                
                if (context.dataset.yAxisID === 'yVolume') {
                  const vol = context.parsed.y;
                  const volStr = vol >= 1000000 
                    ? `${(vol / 1000000).toFixed(2)}M` 
                    : vol >= 1000 
                      ? `${(vol / 1000).toFixed(2)}K` 
                      : vol.toFixed(0);
                  return [`Volume: ${volStr}`];
                }
                
                if (context.dataset.yAxisID === 'yTrends') {
                  return [`Search Interest: ${context.parsed.y.toFixed(0)}/100`];
                }
                
                lines.push(`Price: $${context.parsed.y.toFixed(2)}`);
                
                const date = context.label;
                const negData = negativesByDate.get(date);
                
                if (negData) {
                  lines.push('');
                  lines.push(`⚠️  ${negData.totalCount} Negative Article${negData.totalCount > 1 ? 's' : ''}`);
                  
                  if (negData.ceoCount > 0) {
                    lines.push(`   CEO: ${negData.ceoCount}`);
                  }
                  if (negData.brandCount > 0) {
                    lines.push(`   Brand: ${negData.brandCount}`);
                  }
                  
                  const previewHeadlines = negData.allHeadlines.slice(0, 3);
                  if (previewHeadlines.length > 0) {
                    lines.push('');
                    previewHeadlines.forEach((headline, i) => {
                      const maxLength = window.innerWidth < 768 ? 40 : 55;
                      const truncated = headline.length > maxLength 
                        ? headline.substring(0, maxLength - 3) + '...' 
                        : headline;
                      lines.push(`• ${truncated}`);
                    });
                    
                    if (negData.allHeadlines.length > 3) {
                      lines.push(`   ... and ${negData.allHeadlines.length - 3} more`);
                    }
                  }
                }
                
                return lines;
              }
            }
          }
        },
        scales: {
          yPrice: {
            type: 'linear',
            position: 'left',
            beginAtZero: false,
            ticks: {
              color: '#ebf2f2',
              font: {
                size: window.innerWidth < 768 ? 10 : 11
              },
              callback: function(value) {
                return '$' + value.toFixed(2);
              }
            },
            grid: {
              color: 'rgba(255,255,255,.08)'
            }
          },
          yVolume: {
            type: 'linear',
            position: 'right',
            display: showVolume && stock.volumeHistory && stock.volumeHistory.length > 0,
            beginAtZero: true,
            ticks: {
              color: 'rgba(88, 219, 237, 0.8)',
              font: {
                size: window.innerWidth < 768 ? 9 : 10
              },
              callback: function(value) {
                if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                return value.toFixed(0);
              }
            },
            grid: {
              display: false
            }
          },
          yTrends: {
            type: 'linear',
            position: 'right',
            display: showTrends && trendsData && trendsData.trendsHistory && trendsData.trendsHistory.length > 0,
            beginAtZero: true,
            max: 100,
            ticks: {
              color: 'rgba(168, 85, 247, 0.8)',
              font: {
                size: window.innerWidth < 768 ? 9 : 10
              },
              callback: function(value) {
                return value.toFixed(0);
              }
            },
            grid: {
              display: false
            },
            offset: showVolume
          },
          x: {
            ticks: {
              color: '#ebf2f2',
              font: {
                size: window.innerWidth < 768 ? 9 : 11
              },
              maxTicksLimit: window.innerWidth < 768 ? 6 : 10,
              autoSkip: true
            },
            grid: {
              color: 'rgba(255,255,255,.05)'
            }
          }
        }
      }
    });
    
    const volumeBtn = document.getElementById('volumeToggle');
    if (volumeBtn) {
      if (showVolume) {
        volumeBtn.classList.add('active');
      } else {
        volumeBtn.classList.remove('active');
      }
      
      if (!stock.volumeHistory || stock.volumeHistory.length === 0) {
        volumeBtn.style.display = 'none';
      } else {
        volumeBtn.style.display = 'block';
      }
      
      const newBtn = volumeBtn.cloneNode(true);
      volumeBtn.parentNode.replaceChild(newBtn, volumeBtn);
      
      newBtn.onclick = () => {
        showVolume = !showVolume;
        showStockChart(company);
      };
    }
    
    const trendsBtn = document.getElementById('trendsToggle');
    if (trendsBtn) {
      if (showTrends) {
        trendsBtn.classList.add('active');
      } else {
        trendsBtn.classList.remove('active');
      }
      
      const trendsData = globalTrendsData[company];
      if (!trendsData || !trendsData.trendsHistory || trendsData.trendsHistory.length === 0) {
        trendsBtn.style.display = 'none';
      } else {
        trendsBtn.style.display = 'block';
      }
      
      const newTrendsBtn = trendsBtn.cloneNode(true);
      trendsBtn.parentNode.replaceChild(newTrendsBtn, trendsBtn);
      
      newTrendsBtn.onclick = () => {
        showTrends = !showTrends;
        showStockChart(company);
      };
    }
  }

  function closeStockChart() {
    document.getElementById('stockChartModal').classList.remove('open');
  }

  function clampDateStart(total){
    if (total <= DATE_WINDOW_SIZE) return 0;
    const maxStart = total - DATE_WINDOW_SIZE;
    const start = (dateWindowStart ?? maxStart);
    return Math.min(Math.max(0, start), maxStart);
  }

  function sliceWindow(arr, start, size){ return arr.slice(start, start + size); }

  function updateDateRangeUI(allDates){
    const start = clampDateStart(allDates.length);
    const end = Math.min(allDates.length, start + DATE_WINDOW_SIZE);
    const label = allDates.length ? `${allDates[start]} — ${allDates[end - 1]}` : '';
    document.querySelectorAll('.dates-range').forEach(el => el.textContent = label);

    const atStart = start === 0;
    const atEnd   = (start + DATE_WINDOW_SIZE) >= allDates.length;
    document.querySelectorAll('.dates-prev').forEach(b => b.disabled = atStart);
    document.querySelectorAll('.dates-next').forEach(b => b.disabled = atEnd);
  }
  function hookDatePager(allDates){
    const goPrev = () => {
      const total = allDates.length;
      if (total <= DATE_WINDOW_SIZE) return;
      const start = clampDateStart(total);
      dateWindowPinned = true;
      dateWindowStart = Math.max(0, start - DATE_WINDOW_SIZE);
      renderCharts();
    };
    const goNext = () => {
      const total = allDates.length;
      if (total <= DATE_WINDOW_SIZE) return;
      const maxStart = total - DATE_WINDOW_SIZE;
      dateWindowPinned = true;
      dateWindowStart = Math.min(maxStart, clampDateStart(total) + DATE_WINDOW_SIZE);
      renderCharts();
    };
    document.querySelectorAll('.dates-prev').forEach(b => b.onclick = goPrev);
    document.querySelectorAll('.dates-next').forEach(b => b.onclick = goNext);
  }

  /********************** CSV Fetch with Preload Cache **********************/
  const _preloadCache = new Map();
  const _modalHeadlinesCache = new Map();
  const _modalSerpCache = new Map();
  const _serpFeatureItemsCache = new Map();
  const PERF_LOG = new URLSearchParams(window.location.search).get('perf') === '1';

  async function fetchCsv(url){
    const sharedData = window.CrisisDashboardData;
    if (sharedData?.fetchCsv) {
      return sharedData.fetchCsv(url);
    }
    const t0 = performance.now();
    // Check preload cache first
    if (_preloadCache.has(url)) {
      const cached = _preloadCache.get(url);
      if (cached instanceof Promise) {
        const data = await cached;
        if (PERF_LOG) console.log(`⏱️ ${url} ${(performance.now() - t0).toFixed(0)}ms (cached promise)`);
        return data;
      }
      if (PERF_LOG) console.log(`⏱️ ${url} ${(performance.now() - t0).toFixed(0)}ms (cached)`);
      return cached;
    }
    
    const fetchPromise = (async () => {
      const r = await fetch(url, {cache:'default'});
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
      const contentType = r.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        return await r.json();
      }
      const t = await r.text();
      return await new Promise((res, rej) => {
        Papa.parse(t, {header:true, skipEmptyLines:true,
          complete: out => res(out.data || []), error: e => rej(e)
        });
      });
    })();
    
    _preloadCache.set(url, fetchPromise);
    
    try {
      const data = await fetchPromise;
      _preloadCache.set(url, data);
      if (PERF_LOG) console.log(`⏱️ ${url} ${(performance.now() - t0).toFixed(0)}ms`);
      return data;
    } catch (e) {
      _preloadCache.delete(url);
      if (PERF_LOG) console.log(`⏱️ ${url} ${(performance.now() - t0).toFixed(0)}ms (error)`);
      throw e;
    }
  }

  // Preload modal data in background for faster popup
  function preloadModalData(date) {
    // Modal endpoints now require an entity; skip global preloading.
    console.log(`📦 Skipping modal preload for ${date} (entity required)`);
  }

  async function fetchCsvAny(candidates){
    const sharedData = window.CrisisDashboardData;
    if (sharedData?.fetchCsvAny) {
      return sharedData.fetchCsvAny(candidates);
    }
    for (const u of candidates){
      try { const rows = await fetchCsv(u); if (Array.isArray(rows)) return rows; }
      catch(e){ }
    }
    return [];
  }

  function isTrue(val){
    return ['true','1','yes','y'].includes(String(val || '').trim().toLowerCase());
  }

  async function loadRoster(){
    try{
      const rows = await fetchCsvAny(ROSTER_CANDIDATES);
      rosterMap = new Map();
      companyToCeo = new Map();
      favoriteCeos = new Set();
      favoriteCompanies = new Set();
      fortuneFlags = new Map();
      companySector = new Map();
      companyTickers = new Map();
      rows.forEach(r => {
        const ceo = (r.ceo || r.CEO || '').trim();
        const company = (r.company || r.Company || '').trim();
        const sector = String(r.sector || r.Sector || r.industry || r.Industry || '').trim();
        const fav = isTrue(r.ceo_favorite || r['CEO Favorite'] || r['Favorite CEO'] || r.favorite || r.Favorite);
        const favCompany = isTrue(r.company_favorite || r['Company Favorite'] || r['Favorite Company'] || r.favorite || r.Favorite);
        const ticker = String(r.ticker || r.Ticker || r.stock_ticker || r['Stock Ticker'] || '').trim();
        const f500 = String(r['Fortune 500'] || r.fortune_500 || r.fortune500 || '').trim();
        const f1000 = String(r['Fortune 1000'] || r.fortune_1000 || r.fortune1000 || '').trim();
        const forbes = String(r['Forbes'] || r['Forbes 100'] || r['Forbes 2000'] || r.forbes || r.forbes_100 || r.forbes_2000 || '').trim();
        if (ceo && company){
          rosterMap.set(ceo, company);
          companyToCeo.set(company, ceo);
        }
        if (ceo && company && (fav || favCompany)) favoriteCeos.add(ceo);
        if (company && favCompany) favoriteCompanies.add(company);
        if (company && ticker) companyTickers.set(company, ticker);
        if (company && sector) companySector.set(company, sector);
        if (company) {
          fortuneFlags.set(company, {
            f500: isTrue(f500) || f500.toLowerCase() === 'x',
            f1000: isTrue(f1000) || f1000.toLowerCase() === 'x',
            forbes: isTrue(forbes) || forbes.toLowerCase() === 'x'
          });
        }
      });
      populateSectorFilterOptions();
      setLoadStatus('roster', 'done');
    } catch (e){
      rosterMap = new Map();
      companyToCeo = new Map();
      favoriteCeos = new Set();
      favoriteCompanies = new Set();
      fortuneFlags = new Map();
      companySector = new Map();
      companyTickers = new Map();
      populateSectorFilterOptions();
      setLoadStatus('roster', 'error');
    }
  }

  function populateSectorFilterOptions(){
    const sel = document.getElementById('sectorFilterSelect');
    if (!sel) return;
    const prev = sel.value || '';
    const sectors = [...new Set([...companySector.values()].filter(Boolean))].sort((a,b)=>a.localeCompare(b));
    sel.innerHTML = `<option value="">All sectors</option>` + sectors.map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('');
    if (prev && sectors.includes(prev)) sel.value = prev;
  }

  async function loadBoards(){
    boardsByCeo = new Map();
    try{
      const rows = await fetchCsv(BOARDS_CSV);
      rows.forEach(r => {
        const ceoRaw = (r.ceo || r.Name || '').trim();
        const url = (r.url || r.Website || '').trim();
        let domain = (r.domain || '').trim();

        if (!ceoRaw || !url) return;
        if (!domain && url){
          try { domain = new URL(url).hostname.replace(/^www\./,''); } catch {}
        }

        const key = canonName(ceoRaw);
        const arr = boardsByCeo.get(key) || [];
        arr.push({ domain, url });
        boardsByCeo.set(key, arr);
      });
      setLoadStatus('boards', 'done');
    }catch(e){
      console.error('Could not load boards CSV:', e);
      boardsByCeo = new Map();
      setLoadStatus('boards', 'error');
    }
  }

  async function loadCounts(){
    try{
      const rows = await fetchCsv(COUNTS_PATH());
      allCountsRows = rows
        .map(r => ({
          date: String(r.date || '').trim(),
          ceo: String(r.ceo || '').trim(),
          company: String(r.company || '').trim(),
          theme: String(r.theme || '').trim(),
          pos: +r.positive || 0,
          neu: +r.neutral || 0,
          neg: +r.negative || 0
        }))
        .filter(r => isISODate(r.date) && r.ceo);
      setLoadStatus('news', 'done');
      updateDateOptions();
      maybeRenderAll();
    } catch (e){
      allCountsRows = [];
      setLoadStatus('news', 'error');
    }
  }

  async function loadSerpDaily(){
    try{
      const rows = await fetchCsv(SERPS_DAILY_CSV());
      serpsDaily = rows.map(r => ({
        date: (r.date||'').trim(),
        ceo: (r.ceo||'').trim(),
        company: (r.company||'').trim(),
        total: +r.total || 0,
        neg_serp: +r.negative_serp || +r.neg_serp || 0,
        ctrl: +r.controlled || +r.control || 0,
      })).filter(r => isISODate(r.date));
      setLoadStatus('serps', 'done');
      updateDateOptions();
      maybeRenderAll();
    }catch(e){
      serpsDaily = [];
      setLoadStatus('serps', 'error');
    }
  }

  async function loadSerpFeaturesIndex(){
    try{
      const rows = await fetchCsv(SERP_FEATURES_INDEX_PATH());
      const ctrlRows = await fetchCsv(SERP_FEATURES_CONTROL_INDEX_PATH());
      serpFeatureIndexRows = rows.map(r=>({
        date: String(r.date||'').trim(),
        entity: String(r.entity_name||'').trim(),
        feature: String(r.feature_type||'').trim(),
        total: +r.total_count||0,
        neg: +r.negative_count||0
      })).filter(r=>isISODate(r.date) && r.entity && r.feature);
      serpFeatureControlIndexRows = ctrlRows.map(r=>({
        date: String(r.date||'').trim(),
        entity: String(r.entity_name||'').trim(),
        feature: String(r.feature_type||'').trim(),
        total: +r.total_count||0,
        ctrl: +r.controlled_count||0
      })).filter(r=>isISODate(r.date) && r.entity && r.feature);
      const d = document.getElementById('dateSelect').value;
      if (d) {
        await loadTopStoriesForDate(d);
      }
      setLoadStatus('features', 'done');
      maybeRenderAll();
    }catch(e){
      serpFeatureIndexRows = [];
      serpFeatureControlIndexRows = [];
      topStoriesNegByDate = new Map();
      setLoadStatus('features', 'error');
      maybeRenderAll();
    }
  }

  async function loadTopStoriesForDate(d){
    if (!d) return;
    try{
      const topRows = await fetchCsv(SERP_FEATURES_TOP_STORIES_PATH(d));
      topStoriesNegByDate = new Map();
      topRows.forEach(r=>{
        const date = String(r.date||'').trim();
        const entity = normEntityKey(r.entity_name);
        if (!isISODate(date) || !entity) return;
        const total = +r.total_count||0;
        const neg = +r.negative_count||0;
        topStoriesNegByDate.set(`${date}|${entity}`, total > 0 ? (neg / total) : null);
      });
    }catch(e){
      topStoriesNegByDate = new Map();
    }
  }

  async function loadSerpFeaturesForEntity(entityName){
    if (!entityName) return [];
    if (_serpFeatureCache.has(entityName)) {
      serpFeatureEntityRows = _serpFeatureCache.get(entityName);
      serpFeatureControlEntityRows = _serpFeatureControlCache.get(entityName) || [];
      return serpFeatureEntityRows;
    }
    try{
      const rows = await fetchCsv(SERP_FEATURES_ENTITY_PATH(entityName));
      const ctrlRows = await fetchCsv(SERP_FEATURES_CONTROL_ENTITY_PATH(entityName));
      const parsed = rows.map(r=>({
        date: String(r.date||'').trim(),
        entity: String(r.entity_name||'').trim(),
        feature: String(r.feature_type||'').trim(),
        total: +r.total_count||0,
        neg: +r.negative_count||0
      })).filter(r=>isISODate(r.date) && r.entity && r.feature);
      _serpFeatureCache.set(entityName, parsed);
      serpFeatureEntityRows = parsed;
      const parsedCtrl = ctrlRows.map(r=>({
        date: String(r.date||'').trim(),
        entity: String(r.entity_name||'').trim(),
        feature: String(r.feature_type||'').trim(),
        total: +r.total_count||0,
        ctrl: +r.controlled_count||0
      })).filter(r=>isISODate(r.date) && r.entity && r.feature);
      _serpFeatureControlCache.set(entityName, parsedCtrl);
      serpFeatureControlEntityRows = parsedCtrl;
      return parsed;
    }catch(e){
      serpFeatureEntityRows = [];
      serpFeatureControlEntityRows = [];
      return [];
    }
  }

  function computeRisk(negPct, ctrlPct){
    if (negPct == null || isNaN(negPct)) return 'N/A';
    if (negPct > 0) return 'High';
    if (ctrlPct == null || isNaN(ctrlPct)) return 'N/A';
    return (ctrlPct < 0.35) ? 'Medium' : 'Low';
  }

  function flagIcons(r){
    const edited = !!(r.sentiment_override || r.control_override);
    const llm = !!(r.llm_label);
    if (!edited && !llm) return '';
    const parts = [];
    if (edited) parts.push('<span title="Manually edited">✏️ Manually edited</span>');
    if (llm) parts.push('<span title="AI enriched">🤖 AI enriched</span>');
    return parts.join(' ');
  }

  function toPct01(x){
    if (x == null || x === '') return null;
    const n = +String(x).replace('%','').trim();
    if (!isFinite(n)) return null;
    return n > 1 ? n/100 : n;
  }
  function normEntityKey(val){
    return String(val || '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '')
      .trim();
  }

  async function getArticlesForDate(d){
    if (_articlesCache.has(d)) return _articlesCache.get(d);
    let map = new Map();
    if (!d) return map;
    try{
      const rows = await fetchCsv(ARTICLES_DAILY_PATH(d));
      rows.forEach(r=>{
        const ceo = String(r.ceo||'').trim();
        if (!ceo) return;
        const company = String(r.company||'').trim();
        const theme   = String(r.theme||'').trim();

        let neg = toPct01(r.neg_pct);
        if (neg == null){
          const negCount = +r.negative || 0;
          const tot = (+r.total) || ((+r.positive||0)+(+r.neutral||0)+negCount) || 0;
          neg = tot ? (negCount / tot) : 0;
        }
        map.set(ceo, { company, theme, neg_pct: neg });
      });
    }catch(e){ }
    _articlesCache.set(d, map);
    return map;
  }

  async function getSerpAggForDate(d){
    if (_serpAggCache.has(d)) return _serpAggCache.get(d);
    let map = new Map();
    if (!d) return map;
    try{
      const rows = await fetchCsv(SERP_PROCESSED_PATH(d));
      rows.forEach(r=>{
        const ceo = String(r.ceo||'').trim();
        if (!ceo) return;
        map.set(ceo, {
          total:      +r.total || 0,
          neg_serp:   +r.negative_serp || 0,
          controlled: +r.controlled || 0
        });
      });
    }catch(e){ }
    _serpAggCache.set(d, map);
    return map;
  }

  async function init(){
    initChartOrderPreference();
    initLoadStatus();
    dataReady = false;
    bindLookbackControls();
    updateChartSkeletons();
    const sel = document.getElementById('dateSelect');
    // 2. UI Event Listeners
    const filterInput = document.getElementById('filterInput');
    if (filterInput && filterInput.dataset.boundInput !== '1'){
      filterInput.addEventListener('input', ()=>{ currentPage=1; renderAll().then(() => {
        if (filteredRows.length === 1) selectCeo(filteredRows[0].ceo);
      }); });
      filterInput.dataset.boundInput = '1';
    }
    const sectorFilterSelect = document.getElementById('sectorFilterSelect');
    if (sectorFilterSelect){
      sectorFilterSelect.onchange = ()=>{ currentPage = 1; renderAll(); };
    }
    document.getElementById('refreshBtn').onclick = async ()=>{ await init(); };
    document.getElementById('clearBtn').onclick = ()=>{
      selectedCeo=null;
      document.getElementById('filterInput').value='';
      const sectorFilter = document.getElementById('sectorFilterSelect');
      if (sectorFilter) sectorFilter.value = '';
      updateEntityUrlParam('ceo', null);
      renderAll();
    };
    const companySizeSelect = document.getElementById('companySizeSelect');
    if (companySizeSelect){
      companySizeSelect.value = companySizeFilter;
      companySizeSelect.onchange = ()=>{
        companySizeFilter = companySizeSelect.value || 'all';
        renderAll();
      };
    }
    const crisisBtn = document.getElementById('crisisBtn');
    if (crisisBtn){
      crisisBtn.onclick = async ()=>{
        crisisOnly = !crisisOnly;
        crisisBtn.classList.toggle('active', crisisOnly);
        if (crisisOnly && crisisEntityKeys.size === 0) {
          await loadCrisisEntityKeys();
        }
        renderAll();
      };
    }
    document.getElementById('prevBtn').onclick = ()=>{ if (currentPage>1){ currentPage--; renderTable(); } };
    document.getElementById('nextBtn').onclick = ()=>{ const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE)); if (currentPage<totalPages){ currentPage++; renderTable(); } };

    // 3. Sorting Listeners (Defined ONCE)
    document.querySelectorAll('thead th[data-key]').forEach(th=>{
      th.style.cursor='pointer';
      if (th.dataset.boundSort !== '1'){
        th.addEventListener('click', ()=>{
          const k = th.getAttribute('data-key');
          if (currentSort.key === k) currentSort.dir *= -1; else { currentSort.key = k; currentSort.dir = 1; }
          renderTable();
        });
        th.dataset.boundSort = '1';
      }
    });

    // --- [NEW] AUTO-FILTER FROM URL ---
    // This runs immediately when the dashboard loads
    const urlParams = new URLSearchParams(window.location.search);
    const companyParam = urlParams.get('company'); // e.g. "Nike" or "Tim Cook"
    if (companyParam) {
      const input = document.getElementById('filterInput');
      if (input) {
        input.value = companyParam;
      }
    }
    // ----------------------------------

    // 1. Date Change Listener (bind early)
    if (!sel.dataset.boundChange){
      sel.addEventListener('change', ()=>{ 
        currentPage=1; 
        selectedCeo=null; 
        if (sel.value) {
          loadTopStoriesForDate(sel.value).then(()=>{ renderAll(); });
        }
        renderAll(); 
        if (PRELOAD_MODAL_DATA) {
          setTimeout(()=>{ if (sel.value) preloadModalData(sel.value); }, 150);
        }
      });
      sel.dataset.boundChange = '1';
    }
    const stockChartClose = document.getElementById('stockChartClose');
    if (stockChartClose && stockChartClose.dataset.boundClose !== '1'){
      stockChartClose.addEventListener('click', closeStockChart);
      stockChartClose.dataset.boundClose = '1';
    }

    renderTable();

    const scheduleIdle = window.CrisisDashboardLegacy?.scheduleIdle || ((cb) => setTimeout(cb, 0));
    scheduleIdle(async () => {
      await loadRoster();
      const tasks = [loadCounts(), loadSerpDaily(), loadSerpFeaturesIndex()];
      if (crisisOnly || crisisEntityKeys.size) tasks.push(loadCrisisEntityKeys());
      await Promise.all(tasks);
      loadBoards();
      renderAll();
    });

    scheduleIdle(() => {
      loadStockData().then(data => {
        globalStockData = data;
        if (Object.keys(globalStockData).length > 0) {
          setLoadStatus('stock', 'done');
        }
        renderTable();
        return data;
      }).then(() => {
        console.log('📈 Stock data loaded. Companies with stock data:', Object.keys(globalStockData).length);
        if (Object.keys(globalStockData).length === 0) {
          console.warn('⚠️  No stock data was loaded. Stock prices and sparklines will show N/A.');
        }
      }).catch(()=>{ setLoadStatus('stock','error'); });
      loadTrendsData().then(data => {
        globalTrendsData = data;
        setLoadStatus('trends', 'done');
      }).catch(()=>{ setLoadStatus('trends','error'); });
      loadNegativeSummaryIndex();
    });

    // 4. Modal Close
    
    if (!window.__riskdashCeoStockModalBound){
      window.addEventListener('click', (event) => {
        if (event.target === document.getElementById('stockChartModal')) {
          closeStockChart();
        }
      });
      window.__riskdashCeoStockModalBound = true;
    }
  }

  async function buildRowsForDate(d){
    const articles = await getArticlesForDate(d);
    const serpAgg  = await getSerpAggForDate(d);

    const out = [];
    for (const [ceo, a] of articles.entries()){
      const s = serpAgg.get(ceo);

      let negSerpPct = null, ctrlPct = null;
      if (s && s.total > 0){
        negSerpPct = s.neg_serp   / s.total;
        ctrlPct    = s.controlled / s.total;
      }
      const risk = computeRisk(negSerpPct, ctrlPct);
      const negTopStoriesPct = topStoriesNegByDate.get(`${d}|${normEntityKey(ceo)}`) ?? null;

      const stock = globalStockData[a.company];
      const dailyChange = stock?.dailyChange ?? null;

      out.push({
        date: d,
        ceo,
        company: a.company || '',
        theme: a.theme || 'None',
        neg_news: a.neg_pct ?? 0,
        neg_top_stories: negTopStoriesPct,
        neg_serp: negSerpPct,
        ctrl_pct: ctrlPct,
        daily_change: dailyChange,
        risk
      });
    }
    return out;
  }

  async function renderAll(){
    const d = document.getElementById('dateSelect').value;
    const filter = document.getElementById('filterInput').value.trim().toLowerCase();
    const sectorFilter = (document.getElementById('sectorFilterSelect')?.value || '').trim();

    const rows = (await buildRowsForDate(d)).filter(r=>{
      if (filter && !(r.ceo.toLowerCase().includes(filter) || r.company.toLowerCase().includes(filter))) return false;
      if (sectorFilter && companySector.get(r.company) !== sectorFilter) return false;
      if (!matchesCompanySize(r.company, r.ceo)) return false;
      if (crisisOnly && !crisisEntityKeys.has(normEntityKey(r.ceo))) return false;
      return true;
    });

    filteredRows = rows;
    currentPage = 1;

    if (rows.length === 1) {
      selectedCeo = rows[0].ceo;
      updateEntityUrlParam('ceo', selectedCeo);
    } else if (selectedCeo && !rows.some(r => r.ceo === selectedCeo)) {
      selectedCeo = null;
    }
    if (selectedCeo) {
      await loadSerpFeaturesForEntity(selectedCeo);
    }

    renderTable();
    if (chartsDataReady()) renderCharts();
    _dataRendered = true;
    maybeHideLoadStatus();
  }

  function riskRank(risk) {
    if (risk === 'High') return 3;
    if (risk === 'Medium') return 2;
    if (risk === 'Low') return 1;
    return 0;
  }

  function renderTable(){
    const tbody = document.getElementById('tbody');
    if (!dataReady) {
      tbody.innerHTML = Array.from({length: 6}).map(() => `
        <tr class="skeleton-row">
          <td colspan="13"><div class="skeleton-bar full"></div></td>
        </tr>
      `).join('');
      return;
    }
    let rows = [...filteredRows];

    if (currentSort.key){
      rows.sort((a,b)=>{
        const ka = a[currentSort.key], kb = b[currentSort.key];

        if (currentSort.key === 'risk') {
          return currentSort.dir * (riskRank(ka) - riskRank(kb));
        }

        if (typeof ka === 'string') {
          return currentSort.dir * ka.localeCompare(kb);
        }
        return currentSort.dir * ((ka ?? -1) - (kb ?? -1));
      });
    }

    const start = (currentPage-1)*PAGE_SIZE;
    const pageRows = rows.slice(start, start+PAGE_SIZE);
    
    const rowsHTML = pageRows.map(r=>{
      const checked = (selectedCeo && selectedCeo===r.ceo) ? 'checked' : '';
      const stock = globalStockData[r.company];
      const ticker = companyTickers.get(r.company) || '';
      const hasTicker = Boolean(ticker);
      const favActive = favoriteCeos.has(r.ceo);
      const favStar = `<span class="fav-inline ${favActive ? 'active' : ''}" title="Favorite">★</span>`;
      let stockPriceHtml = '<span class="muted">N/A</span>';
      let dailyChangeHtml = '<span class="muted">N/A</span>';
      let sparklineCellHtml = '<span class="muted">N/A</span>';
      
      if (stock && stock.openingPrice !== null) {
        stockPriceHtml = `<span class="stock-price">$${stock.openingPrice.toFixed(2)}</span>`;
        
        if (stock.dailyChange !== null) {
          const changeClass = stock.dailyChange >= 0 ? 'positive' : 'negative';
          const changeSymbol = stock.dailyChange >= 0 ? '▲' : '▼';
          dailyChangeHtml = `<span class="stock-change ${changeClass}">${changeSymbol} ${Math.abs(stock.dailyChange).toFixed(2)}%</span>`;
        }
        
        const sparklineId = `sparkline-${r.company.replace(/[^a-zA-Z0-9]/g, '-')}`;
        sparklineCellHtml = `<span id="${sparklineId}"></span>`;
      } else if (!hasTicker) {
        sparklineCellHtml = '<span class="muted">Privately Owned</span>';
      } else {
        sparklineCellHtml = '<span class="muted">Stock data unavailable</span>';
      }
      
      return `<tr data-company="${esc(r.company)}">
        <td>
          <div class="ceo-cell-content">
            <input type="checkbox" data-ceo="${esc(r.ceo)}" class="brandCheck" ${checked} onclick="event.stopPropagation()" />
            ${favStar}
            <div class="ceo-name-block">
              <div class="ceo-name">${esc(r.ceo)}</div>
              <div class="company-subheader">${esc(r.company)}</div>
            </div>
          </div>
        </td>
        <td>${stockPriceHtml}</td>
        <td>${dailyChangeHtml}</td>
        <td class="sparkline-cell">${sparklineCellHtml}</td>
        <td>${fmtPct(r.neg_news)}</td>
        <td>${fmtPct(r.neg_top_stories)}</td>
        <td>${fmtPct(r.neg_serp)}</td>
        <td>${fmtPct(r.ctrl_pct)}</td>
        <td>${
          (r.risk==='High'||r.risk==='Medium'||r.risk==='Low')
            ? (r.risk==='High'
                ? '<span class="pill high">High</span>'
                : r.risk==='Medium'
                  ? '<span class="pill med">Medium</span>'
                  : '<span class="pill low">Low</span>')
            : '<span class="muted">N/A</span>'
        }</td>
        <td class="action-col"><button class="coverageBtn" data-ceo="${esc(r.ceo)}" data-company="${esc(r.company)}" onclick="event.stopPropagation()">View</button></td>
        <td class="action-col"><button class="boardsBtn" data-ceo="${esc(r.ceo)}" onclick="event.stopPropagation()">View</button></td>
      </tr>`;
    }).join('');
    
    tbody.innerHTML = rowsHTML;
    
    pageRows.forEach(r => {
      const stock = globalStockData[r.company];
      if (stock && stock.priceHistory && stock.priceHistory.length >= 2) {
        const sparklineId = `sparkline-${r.company.replace(/[^a-zA-Z0-9]/g, '-')}`;
        const container = document.getElementById(sparklineId);
        if (container) {
          const firstPrice = stock.priceHistory[0];
          const lastPrice = stock.priceHistory[stock.priceHistory.length - 1];
          const isPositive = lastPrice >= firstPrice;
          container.innerHTML = '';
          const canvas = createSparkline(stock.priceHistory, isPositive);
          if (canvas) container.appendChild(canvas);
          
          const sparklineCell = container.closest('td');
          if (sparklineCell) {
            sparklineCell.onclick = (e) => {
              e.stopPropagation();
              showStockChart(r.company);
            };
          }
        }
      }
    });

    tbody.querySelectorAll('.brandCheck').forEach(cb=>{
      cb.addEventListener('change', async (e)=>{
        const ceo = e.target.getAttribute('data-ceo');
        if (e.target.checked){ selectedCeo = ceo; tbody.querySelectorAll('.brandCheck').forEach(x=>{ if (x!==e.target) x.checked=false; }); }
        else selectedCeo = null;
    if (selectedCeo) updateEntityUrlParam('ceo', selectedCeo);
        if (selectedCeo) {
          await loadSerpFeaturesForEntity(selectedCeo);
        }
        if (chartsDataReady()) renderCharts();
      });
    });
    
    tbody.querySelectorAll('.coverageBtn').forEach(btn=>{
      btn.onclick = async ()=>{
        const ceo = btn.getAttribute('data-ceo');
        const company = btn.getAttribute('data-company');
        await selectCeo(ceo);
        showHeadlines({ceo, company});
      };
    });
    
    tbody.querySelectorAll('.boardsBtn').forEach(btn=>{
      btn.onclick = ()=>{
        const ceo = btn.getAttribute('data-ceo');
        showBoards(ceo);
      };
    });

    document.getElementById('pageNo').textContent = String(currentPage);
    document.getElementById('pageTotal').textContent = String(Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE)));
  }

  function getDateSeries(){
    const byDateAll = new Map();
    const byDateSel = new Map();
    allCountsRows.forEach(r=>{
      const key = r.date;
      if (!byDateAll.has(key)) byDateAll.set(key,{pos:0,neu:0,neg:0});
      const all = byDateAll.get(key);
      all.pos += +r.pos||0; all.neu += +r.neu||0; all.neg += +r.neg||0;
      if (!selectedCeo || r.ceo===selectedCeo){
        if (!byDateSel.has(key)) byDateSel.set(key,{pos:0,neu:0,neg:0});
        const bucket = byDateSel.get(key);
        bucket.pos += +r.pos||0; bucket.neu += +r.neu||0; bucket.neg += +r.neg||0;
      }
    });

    const serpByDateAll = new Map();
    const serpByDateSel = new Map();
    serpsDaily.forEach(r=>{
      const key = r.date;
      if (!serpByDateAll.has(key)) serpByDateAll.set(key,{total:0,neg:0,ctrl:0});
      const all = serpByDateAll.get(key);
      all.total += +r.total||0; all.neg += +r.neg_serp||0; all.ctrl += +r.ctrl||0;
      if (!selectedCeo || r.ceo===selectedCeo){
        if (!serpByDateSel.has(key)) serpByDateSel.set(key,{total:0,neg:0,ctrl:0});
        const b = serpByDateSel.get(key);
        b.total += +r.total||0; b.neg += +r.neg_serp||0; b.ctrl += +r.ctrl||0;
      }
    });

    const dates = [...new Set([...byDateAll.keys(), ...serpByDateAll.keys()])].filter(isISODate).sort();
    const newsPos=[], newsNeu=[], newsNeg=[], serpNegPct=[], serpCtrlPct=[], serpNegPctAll=[], serpCtrlPctAll=[];
    dates.forEach(d=>{
      const n = byDateSel.get(d)||{pos:0,neu:0,neg:0};
      const t = n.pos+n.neu+n.neg || 0;
      newsPos.push(t? n.pos/t*100 : 0);
      newsNeu.push(t? n.neu/t*100 : 0);
      newsNeg.push(t? n.neg/t*100 : 0);

      const s = serpByDateSel.get(d)||{total:0,neg:0,ctrl:0};
      serpNegPct.push(s.total? s.neg/s.total*100 : 0);
      serpCtrlPct.push(s.total? s.ctrl/s.total*100 : 0);

      const sa = serpByDateAll.get(d)||{total:0,neg:0,ctrl:0};
      serpNegPctAll.push(sa.total? sa.neg/sa.total*100 : 0);
      serpCtrlPctAll.push(sa.total? sa.ctrl/sa.total*100 : 0);
    });
    return {dates, newsPos, newsNeu, newsNeg, serpNegPct, serpCtrlPct, serpNegPctAll, serpCtrlPctAll};
  }

  function getFeatureSeries(){
    const byDate = new Map();
    const totalByDate = new Map();
    const rows = selectedCeo ? serpFeatureEntityRows : serpFeatureIndexRows;
    const hasSentiment = rows.some(r => isItemFeature(r.feature));
    let order = hasSentiment ? FEATURE_ORDER_SENTIMENT : FEATURE_ORDER_BASE;
    const organicByDate = new Map();
    serpsDaily.forEach(r=>{
      if (selectedCeo && r.ceo !== selectedCeo) return;
      const key = r.date;
      if (!organicByDate.has(key)) organicByDate.set(key, {total:0, neg:0});
      const o = organicByDate.get(key);
      o.total += +r.total||0;
      o.neg += +r.neg_serp||0;
    });
    rows.forEach(r=>{
      if (!isItemFeature(r.feature)) return;
      const feature = r.feature;
      if (!totalByDate.has(r.date)) totalByDate.set(r.date, 0);
      totalByDate.set(r.date, totalByDate.get(r.date) + (r.total || 0));
      if (!byDate.has(r.date)) byDate.set(r.date, {});
      const bucket = byDate.get(r.date);
      if (!bucket[feature]) bucket[feature] = {neg:0};
      bucket[feature].neg += r.neg || 0;
    });
    organicByDate.forEach((o, date)=>{
      if (!totalByDate.has(date)) totalByDate.set(date, 0);
      totalByDate.set(date, totalByDate.get(date) + o.total);
      if (!byDate.has(date)) byDate.set(date, {});
      const bucket = byDate.get(date);
      if (!bucket.organic) bucket.organic = {neg:0};
      bucket.organic.neg += o.neg || 0;
    });
    const dates = [...totalByDate.keys()].filter(isISODate).sort();
    const series = {};
    order.forEach(f => { series[f] = []; });
    dates.forEach(d=>{
      const bucket = byDate.get(d) || {};
      const dayTotal = totalByDate.get(d) || 0;
      order.forEach(f=>{
        const v = bucket[f] || {neg:0};
        series[f].push(dayTotal ? (v.neg / dayTotal * 100) : 0);
      });
    });
    return {dates, series, order, countsByDate: byDate, totalsByDate: totalByDate};
  }

  function getFeatureControlSeries(){
    const byDate = new Map();
    const totalByDate = new Map();
    const rows = selectedCeo ? serpFeatureControlEntityRows : serpFeatureControlIndexRows;
    const order = FEATURE_ORDER_SENTIMENT;
    const organicByDate = new Map();
    serpsDaily.forEach(r=>{
      if (selectedCeo && r.ceo !== selectedCeo) return;
      const key = r.date;
      if (!organicByDate.has(key)) organicByDate.set(key, {total:0, ctrl:0});
      const o = organicByDate.get(key);
      o.total += +r.total||0;
      o.ctrl += +r.ctrl||0;
    });
    rows.forEach(r=>{
      if (!isItemFeature(r.feature)) return;
      const feature = r.feature;
      if (!byDate.has(r.date)) byDate.set(r.date, {});
      const bucket = byDate.get(r.date);
      if (!bucket[feature]) bucket[feature] = {ctrl:0};
      bucket[feature].ctrl += r.ctrl || 0;
      if (!totalByDate.has(r.date)) totalByDate.set(r.date, 0);
      totalByDate.set(r.date, totalByDate.get(r.date) + (r.total || 0));
    });
    organicByDate.forEach((o, date)=>{
      if (!byDate.has(date)) byDate.set(date, {});
      const bucket = byDate.get(date);
      if (!bucket.organic) bucket.organic = {ctrl:0};
      bucket.organic.ctrl += o.ctrl || 0;
      if (!totalByDate.has(date)) totalByDate.set(date, 0);
      totalByDate.set(date, totalByDate.get(date) + o.total);
    });
    const dates = [...totalByDate.keys()].filter(isISODate).sort();
    const series = {};
    order.forEach(f => { series[f] = []; });
    dates.forEach(d=>{
      const bucket = byDate.get(d) || {};
      const dayTotal = totalByDate.get(d) || 0;
      order.forEach(f=>{
        const v = bucket[f] || {ctrl:0};
        series[f].push(dayTotal ? (v.ctrl / dayTotal * 100) : 0);
      });
    });
    return {dates, series, order, countsByDate: byDate, totalsByDate: totalByDate};
  }

  function renderCharts(){
    const {dates, newsPos, newsNeu, newsNeg, serpNegPct, serpCtrlPct, serpNegPctAll, serpCtrlPctAll} = getDateSeries();
    const {dates: featureDates, series: featureSeries, order: featureOrder, countsByDate: featureCountsByDate} = getFeatureSeries();
    const {dates: featureCtrlDates, series: featureCtrlSeries, order: featureCtrlOrder, countsByDate: featureCtrlCountsByDate} = getFeatureControlSeries();

    const maxStart = Math.max(0, dates.length - DATE_WINDOW_SIZE);
    if (!dateWindowPinned || dateWindowStart === null){
      dateWindowStart = maxStart;
    }
    const start = clampDateStart(dates.length);
    const d  = sliceWindow(dates,       start, DATE_WINDOW_SIZE);
    const nP = sliceWindow(newsPos,     start, DATE_WINDOW_SIZE);
    const nN = sliceWindow(newsNeu,     start, DATE_WINDOW_SIZE);
    const nG = sliceWindow(newsNeg,     start, DATE_WINDOW_SIZE);
    const sN = sliceWindow(serpNegPct,  start, DATE_WINDOW_SIZE);
    const sC = sliceWindow(serpCtrlPct, start, DATE_WINDOW_SIZE);
    const sNAll = sliceWindow(serpNegPctAll, start, DATE_WINDOW_SIZE);
    const sCAll = sliceWindow(serpCtrlPctAll, start, DATE_WINDOW_SIZE);
    const featureIndex = new Map(featureDates.map((d,i)=>[d,i]));
    const fDates = d;
    const featureData = featureOrder.map(f=>{
      const arr = featureSeries[f] || [];
      return fDates.map(dt=>{
        const idx = featureIndex.get(dt);
        return idx == null ? 0 : (arr[idx] || 0);
      });
    });
    const featureCtrlIndex = new Map(featureCtrlDates.map((d,i)=>[d,i]));
    const fcDates = d;
    const featureCtrlData = featureCtrlOrder.map(f=>{
      const arr = featureCtrlSeries[f] || [];
      return fcDates.map(dt=>{
        const idx = featureCtrlIndex.get(dt);
        return idx == null ? 0 : (arr[idx] || 0);
      });
    });

    updateDateRangeUI(dates);
    hookDatePager(dates);

    const who = selectedCeo ? selectedCeo : "Index Average";
    const fmtPctInt = v => `${Math.round(v)}%`;
    const pct = fmtPctInt;
    const getClickedDate = (event, chart) => {
      if (!chart) return '';
      const points = chart.getElementsAtEventForMode(event, 'index', { intersect: false }, false);
      const idx = points && points.length ? points[0].index : null;
      if (idx == null) return '';
      return String(chart.data?.labels?.[idx] || '').trim();
    };
    const openChartDetailAt = (kind, date) => {
      if (!isISODate(date)) return;
      const ctx = normalizeCeoModalContext({ ceo: selectedCeo });
      if (!ctx.ceo) {
        openModal('Coverage Detail', '<div class="muted">Select a CEO row first, then click a chart data point.</div>');
        return;
      }
      if (kind === 'news') return void showHeadlines({ ceo: ctx.ceo, company: ctx.company }, date);
      if (kind === 'serp') return void showSerp({ ceo: ctx.ceo, company: ctx.company }, date);
      if (kind === 'features') return void showSerpFeatures({ ceo: ctx.ceo, company: ctx.company }, date);
    };

    const commonOpts = {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { bottom: 24 } },
      scales: {
        x: { ticks: { color: '#ebf2f2' }, grid: { color: 'transparent' } },
        y: {
          ticks: { color: '#ebf2f2', stepSize: 20, callback: (v) => fmtPctInt(v) },
          grid: { color: 'transparent' },
          min: 0, max: 100
        }
      },
      plugins: {
        legend: { labels: { color: '#ebf2f2' } },
        title: { display: true, text: who, color: '#ebf2f2', font: { weight: 'bold', size: 14 }, padding: { top: 10, bottom: 6 } },
        tooltip: { callbacks: { label: (ctx) => {
          const label = ctx.dataset?.label ? `${ctx.dataset.label}: ` : '';
          const val = (ctx.parsed && typeof ctx.parsed.y === 'number') ? ctx.parsed.y : (typeof ctx.parsed === 'number' ? ctx.parsed : 0);
          return `${label}${fmtPctInt(val)}`;
        }}}
      }
    };

    const compositeDatasets = featureOrder.map((feature, idx) => ({
      label: FEATURE_LABELS[feature] || feature,
      data: (featureData[idx] || []).map(v => Math.max(0, Math.min(100, Number(v) || 0))),
      tension: 0.2,
      fill: true,
      borderWidth: 1,
      borderColor: FEATURE_COLORS[idx % FEATURE_COLORS.length],
      backgroundColor: (FEATURE_COLORS[idx % FEATURE_COLORS.length] + 'b3'),
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHitRadius: 10,
      stack: 'composite_features'
    }));
    const newsNegComposite = nG.map(v => -Math.max(0, Math.min(100, Number(v) || 0)));
    const splitFillPlugin = {
      id: 'compositeSplitFill',
      afterDatasetsDraw(chart){
        const {ctx, chartArea, scales} = chart;
        if (!ctx || !chartArea || !scales?.y) return;
        const {left, right} = chartArea;
        const mid = scales.y.getPixelForValue(0);
        const cardBg = getComputedStyle(document.documentElement).getPropertyValue('--card').trim() || '#092e37';
        ctx.save();
        ctx.fillStyle = cardBg;
        ctx.fillRect(left, mid - 2, right - left, 4);
        ctx.restore();
      }
    };
    const compositeCtx = document.getElementById('negativeCompositeChart')?.getContext('2d');
    if (negativeCompositeChart) negativeCompositeChart.destroy();
    if (compositeCtx) {
      negativeCompositeChart = new Chart(compositeCtx, {
        type: 'line',
        data: {
          labels: d,
          datasets: [
            ...compositeDatasets,
            {
              type: 'bar',
              label: 'Negative Google Newsfeed',
              data: newsNegComposite,
              borderWidth: 1,
              borderColor: '#ff8261',
              backgroundColor: 'rgba(255,130,97,.92)',
              isNewsComposite: true,
              order: 10
            }
          ]
        },
        options: {
          ...commonOpts,
          scales: {
            ...commonOpts.scales,
            x: { ...commonOpts.scales.x, stacked: true },
            y: {
              ...commonOpts.scales.y,
              stacked: true,
              min: -100,
              max: 100,
              ticks: {
                ...commonOpts.scales.y.ticks,
                stepSize: 20,
                autoSkip: false,
                maxTicksLimit: 11,
                callback: v => pct(Math.abs(Number(v) || 0))
                },
              grid: {
                ...commonOpts.scales.y.grid,
                color: 'transparent',
                lineWidth: 0
              }
            }
          },
          interaction: { mode: 'index', intersect: false },
          onClick: (event, _elements, chart) => {
            const date = getClickedDate(event, chart);
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);
            const datasetIdx = points && points.length ? points[0].datasetIndex : -1;
            if (datasetIdx === featureOrder.length) return void openChartDetailAt('news', date);
            return void openChartDetailAt('features', date);
          },
          plugins: {
            ...commonOpts.plugins,
            title: { ...commonOpts.plugins.title, text: `${who} • Negative Signal Composite` },
            tooltip: {
              ...commonOpts.plugins.tooltip,
              mode: 'index',
              intersect: false,
              callbacks: {
                ...commonOpts.plugins.tooltip.callbacks,
                label: (ctx) => {
                  const raw = (typeof ctx.parsed?.y === 'number') ? ctx.parsed.y : ctx.parsed;
                  const val = Math.abs(Number(raw) || 0);
                  const date = ctx.label;
                  if (ctx.dataset?.isNewsComposite) {
                    return `${ctx.dataset?.label ? ctx.dataset.label + ': ' : ''}${pct(val)}`;
                  }
                  const feature = featureOrder[ctx.datasetIndex];
                  const bucket = featureCountsByDate.get(date) || {};
                  const count = bucket[feature]?.neg || 0;
                  return `${ctx.dataset?.label ? ctx.dataset.label + ': ' : ''}${pct(val)} (${Math.round(count)})`;
                },
                footer: (items) => {
                  const featureTotal = items
                    .filter(i => !i.dataset?.isNewsComposite)
                    .reduce((sum, i) => sum + Math.abs(Number(i.parsed?.y || 0)), 0);
                  const newsItem = items.find(i => i.dataset?.isNewsComposite);
                  const newsVal = newsItem ? Math.abs(Number(newsItem.parsed?.y || 0)) : 0;
                  return `SERP feature negative: ${Math.round(featureTotal)}% | News negative: ${Math.round(newsVal)}%`;
                }
              }
            }
          }
        },
        plugins: [splitFillPlugin]
      });
    }

    const nh = document.getElementById('newsChart').getContext('2d');
    if (newsChart) newsChart.destroy();
    newsChart = new Chart(nh, {
      type: 'bar',
      data: {
        labels: d,
        datasets: [
          { label: 'Positive %', data: nP, backgroundColor: '#82c618', stack: 's' },
          { label: 'Neutral %',  data: nN, backgroundColor: '#cfdbdd', stack: 's' },
          { label: 'Negative %', data: nG, backgroundColor: '#ff8261', stack: 's' }
        ]
      },
      options: {
        ...commonOpts,
        onClick: (event, _elements, chart) => {
          openChartDetailAt('news', getClickedDate(event, chart));
        }
      }
    });

    const avg = (arr) => {
      const vals = arr.filter(v => Number.isFinite(v));
      if (!vals.length) return null;
      return vals.reduce((sum, v) => sum + v, 0) / vals.length;
    };
    const avgNegAll  = avg(sNAll);
    const avgCtrlAll = avg(sCAll);
    const avgNegLine  = avgNegAll == null ? [] : new Array(d.length).fill(avgNegAll);
    const avgCtrlLine = avgCtrlAll == null ? [] : new Array(d.length).fill(avgCtrlAll);
    
    const SOLID_WIDTH  = 3;
    const DASH_WIDTH   = 2;
    const DASH_PATTERN = [8, 6];
    const NEG_COLOR    = '#ff8261';
    const CTRL_COLOR   = '#58dbed';

    const sh = document.getElementById('serpChart').getContext('2d');
    if (serpChart) serpChart.destroy();
    serpChart = new Chart(sh, {
      type: 'line',
      data: {
        labels: d,
        datasets: [
          {
            label: 'Daily Negative Organic %',
            data: sN,
            tension: 0.25,
            borderWidth: SOLID_WIDTH,
            borderColor: NEG_COLOR,
            fill: false
          },
          {
            label: 'Daily SERP Control %',
            data: sC,
            tension: 0.25,
            borderWidth: SOLID_WIDTH,
            borderColor: CTRL_COLOR,
            fill: false
          },
          {
            label: 'Average Negative Organic %',
            data: avgNegLine,
            tension: 0,
            fill: false,
            borderWidth: DASH_WIDTH,
            borderDash: DASH_PATTERN,
            borderColor: NEG_COLOR,
            pointRadius: 0
          },
          {
            label: 'Average SERP Control %',
            data: avgCtrlLine,
            tension: 0,
            fill: false,
            borderWidth: DASH_WIDTH,
            borderDash: DASH_PATTERN,
            borderColor: CTRL_COLOR,
            pointRadius: 0
          }
        ]
      },
      options: {
        ...commonOpts,
        onClick: (event, _elements, chart) => {
          openChartDetailAt('serp', getClickedDate(event, chart));
        }
      }
    });

    const fh = document.getElementById('featureChart').getContext('2d');
    if (featureChart) featureChart.destroy();
    featureChart = new Chart(fh, {
      type: 'line',
      data: {
        labels: fDates,
        datasets: featureOrder.map((feature, idx) => ({
          label: FEATURE_LABELS[feature] || feature,
          data: featureData[idx],
          tension: 0.2,
          fill: true,
          borderWidth: 1,
          borderColor: FEATURE_COLORS[idx],
          backgroundColor: FEATURE_COLORS[idx] + 'b3',
          pointRadius: 0,
          stack: 'features'
        }))
      },
      options: {
        ...commonOpts,
        scales: {
          ...commonOpts.scales,
          y: { ...commonOpts.scales.y, stacked: true },
          x: { ...commonOpts.scales.x, stacked: true }
        },
        interaction: { mode: 'index', intersect: false },
        onClick: (event, _elements, chart) => {
          openChartDetailAt('features', getClickedDate(event, chart));
        },
        plugins: {
          ...commonOpts.plugins,
          tooltip: {
            ...commonOpts.plugins.tooltip,
            mode: 'index',
            intersect: false,
            callbacks: {
              ...commonOpts.plugins.tooltip.callbacks,
              label: (ctx) => {
                const pctVal = (typeof ctx.parsed.y === 'number') ? ctx.parsed.y : ctx.parsed;
                const date = ctx.label;
                const feature = featureOrder[ctx.datasetIndex];
                const bucket = featureCountsByDate.get(date) || {};
                const count = bucket[feature]?.neg || 0;
                return `${ctx.dataset?.label ? ctx.dataset.label + ': ' : ''}${pct(pctVal)} (${Math.round(count)})`;
              },
              footer: (items) => {
                const total = items.reduce((sum, i) => sum + (i.parsed?.y || 0), 0);
                const date = items[0]?.label;
                const bucket = date ? (featureCountsByDate.get(date) || {}) : {};
                const countTotal = Object.values(bucket).reduce((sum, v) => sum + ((v && v.neg) || 0), 0);
                return `Total negative: ${Math.round(total)}% (${Math.round(countTotal)})`;
              }
            }
          },
          title: { ...commonOpts.plugins.title, text: `${who} • SERP Feature Negative Share` }
        }
      }
    });

    const fch = document.getElementById('featureControlChart').getContext('2d');
    if (featureControlChart) featureControlChart.destroy();
    featureControlChart = new Chart(fch, {
      type: 'line',
      data: {
        labels: fcDates,
        datasets: featureCtrlOrder.map((feature, idx) => ({
          label: FEATURE_LABELS[feature] || feature,
          data: featureCtrlData[idx],
          tension: 0.2,
          fill: true,
          borderWidth: 1,
          borderColor: CONTROL_COLORS[idx % CONTROL_COLORS.length],
          backgroundColor: (CONTROL_COLORS[idx % CONTROL_COLORS.length] + 'b3'),
          pointRadius: 0,
          stack: 'features'
        }))
      },
      options: {
        ...commonOpts,
        scales: {
          ...commonOpts.scales,
          y: { ...commonOpts.scales.y, stacked: true },
          x: { ...commonOpts.scales.x, stacked: true }
        },
        interaction: { mode: 'index', intersect: false },
        onClick: (event, _elements, chart) => {
          openChartDetailAt('features', getClickedDate(event, chart));
        },
        plugins: {
          ...commonOpts.plugins,
          tooltip: {
            ...commonOpts.plugins.tooltip,
            mode: 'index',
            intersect: false,
            callbacks: {
              ...commonOpts.plugins.tooltip.callbacks,
              label: (ctx) => {
                const pctVal = (typeof ctx.parsed.y === 'number') ? ctx.parsed.y : ctx.parsed;
                const date = ctx.label;
                const feature = featureCtrlOrder[ctx.datasetIndex];
                const bucket = featureCtrlCountsByDate.get(date) || {};
                const count = bucket[feature]?.ctrl || 0;
                return `${ctx.dataset?.label ? ctx.dataset.label + ': ' : ''}${pct(pctVal)} (${Math.round(count)})`;
              },
              footer: (items) => {
                const total = items.reduce((sum, i) => sum + (i.parsed?.y || 0), 0);
                const date = items[0]?.label;
                const bucket = date ? (featureCtrlCountsByDate.get(date) || {}) : {};
                const countTotal = Object.values(bucket).reduce((sum, v) => sum + ((v && v.ctrl) || 0), 0);
                return `Total control: ${Math.round(total)}% (${Math.round(countTotal)})`;
              }
            }
          },
          title: { ...commonOpts.plugins.title, text: `${who} • SERP Feature Control Share` }
        }
      }
    });
  }

  const modal = document.getElementById('modal');
  const modalTitle = document.getElementById('modalTitle');
  const modalContent = document.getElementById('modalContent');
  let modalDateOverride = null;
  let modalEntityState = null;
  function normalizeCeoModalContext(input){
    const ctx = (input && typeof input === 'object') ? input : { ceo: String(input || '').trim() };
    const ceo = String(ctx.ceo || '').trim();
    let company = String(ctx.company || '').trim();
    if (!company && ceo) {
      const match = filteredRows.find(r => String(r.ceo || '').trim() === ceo);
      if (match) company = String(match.company || '').trim();
    }
    if (!company && ceo) {
      company = String(rosterMap.get(ceo) || '').trim();
    }
    return { ...ctx, ceo, company };
  }
  function setCeoModalState(input, activeTab){
    modalEntityState = {
      type: 'ceo',
      context: normalizeCeoModalContext(input),
      activeTab: String(activeTab || 'headlines')
    };
  }
  function renderModalTabs(){
    if (!modalEntityState || modalEntityState.type !== 'ceo' || !modalEntityState.context?.ceo) return '';
    const tabs = [
      ['headlines', 'Google News Headlines'],
      ['serp', 'Organic SERP'],
      ['features', 'SERP Features']
    ];
    return `<div class="modal-tabs">${
      tabs.map(([key, label]) =>
        `<button type="button" class="modal-tab${modalEntityState.activeTab===key ? ' active' : ''}" data-modal-tab="${key}">${label}</button>`
      ).join('')
    }</div>`;
  }
  function bindModalTabs(){
    if (!modalEntityState || modalEntityState.type !== 'ceo' || !modalEntityState.context?.ceo) return;
    const ctx = modalEntityState.context;
    modalContent.querySelectorAll('[data-modal-tab]').forEach(btn=>{
      btn.onclick = ()=>{
        const tab = String(btn.getAttribute('data-modal-tab') || '').trim();
        if (!tab || tab === modalEntityState.activeTab) return;
        const d = getModalDate();
        if (tab === 'headlines') return void showHeadlines({ ceo: ctx.ceo, company: ctx.company }, d);
        if (tab === 'serp') return void showSerp({ ceo: ctx.ceo, company: ctx.company }, d);
        if (tab === 'features') return void showSerpFeatures({ ceo: ctx.ceo, company: ctx.company }, d);
      };
    });
  }
  const getDashboardDate = () => String(document.getElementById('dateSelect')?.value || '').trim();
  function syncModalDateOverride(contextDate = null){
    const target = String(contextDate || '').trim();
    if (!isISODate(target)) return;
    const dashboardDate = getDashboardDate();
    modalDateOverride = (target && target !== dashboardDate) ? target : null;
  }
  function getModalDate(contextDate = null){
    if (isISODate(modalDateOverride)) return modalDateOverride;
    const fallback = String(contextDate || getDashboardDate() || '').trim();
    return isISODate(fallback) ? fallback : '';
  }
  function getModalDateOptionsHtml(selectedDate){
    const sel = document.getElementById('dateSelect');
    if (!sel) return '';
    return Array.from(sel.options || []).map(opt => {
      const value = String(opt.value || '').trim();
      if (!isISODate(value)) return '';
      const selected = value === selectedDate ? ' selected' : '';
      return `<option value="${esc(value)}"${selected}>${esc(value)}</option>`;
    }).join('');
  }
  function closeModal(){
    modal.classList.remove('open');
    modalDateOverride = null;
    modalEntityState = null;
  }
  document.getElementById('modalClose').onclick = closeModal;
  modal.addEventListener('click', e=>{ if (e.target===modal) closeModal(); });

  function openModal(title, html, opts = null){
    const options = (opts && typeof opts === 'object' && !Array.isArray(opts)) ? opts : { contextDate: opts };
    const dateText = getModalDate(options.contextDate || null);
    const onDateChange = typeof options.onDateChange === 'function' ? options.onDateChange : null;
    const dateOptions = getModalDateOptionsHtml(dateText);
    const dashboardDate = getDashboardDate();
    const isSynced = !modalDateOverride || modalDateOverride === dashboardDate;
    const datePanel = onDateChange && dateOptions && isISODate(dateText)
      ? `<div class="modal-date-panel">
          <label for="modalDateSelect">Date</label>
          <select id="modalDateSelect">${dateOptions}</select>
          <button id="modalDateReset" class="ghost-btn"${isSynced ? ' disabled' : ''}>Use dashboard date</button>
          <span class="modal-date-status">${isSynced ? 'Synced to dashboard date' : 'Local modal date'}</span>
        </div>`
      : (isISODate(dateText) ? `<div class="modal-date-panel"><span class="modal-date-status">Date: ${dateText}</span></div>` : '');
    const tabsHtml = renderModalTabs();
    const toolbarHtml = (tabsHtml || datePanel) ? `<div class="modal-toolbar">${tabsHtml || ''}${datePanel || ''}</div>` : '';
    const panelHtml = tabsHtml ? `<div class="modal-tab-panel">${html}</div>` : html;
    modalTitle.textContent = title;
    modalContent.innerHTML = `${toolbarHtml}${panelHtml}`;
    modal.classList.add('open');
    bindModalTabs();
    if (onDateChange && dateOptions && isISODate(dateText)){
      const modalDateSel = modalContent.querySelector('#modalDateSelect');
      const resetBtn = modalContent.querySelector('#modalDateReset');
      if (modalDateSel){
        modalDateSel.onchange = ()=>{
          const selected = String(modalDateSel.value || '').trim();
          modalDateOverride = (isISODate(selected) && selected !== dashboardDate) ? selected : null;
          Promise.resolve(onDateChange(getModalDate(selected))).catch(()=>{});
        };
      }
      if (resetBtn){
        resetBtn.onclick = ()=>{
          modalDateOverride = null;
          Promise.resolve(onDateChange(getModalDate())).catch(()=>{});
        };
      }
    }
  }

  function computeModalStats(rows, controlValueFn){
    let pos = 0;
    let neu = 0;
    let neg = 0;
    let ctrl = 0;
    rows.forEach(r=>{
      const s = String(r.sentiment || '').toLowerCase();
      if (s === 'positive') pos += 1;
      else if (s === 'negative') neg += 1;
      else neu += 1;
      if (controlValueFn(r)) ctrl += 1;
    });
    return { pos, neu, neg, ctrl, total: rows.length };
  }

  function renderModalInsightDonuts(sentCanvasId, ctrlCanvasId, stats){
    if (!stats || !stats.total) return;
    const sentCanvas = modalContent.querySelector(`#${sentCanvasId}`);
    const ctrlCanvas = modalContent.querySelector(`#${ctrlCanvasId}`);
    if (!sentCanvas || !ctrlCanvas) return;
    const sentCtx = sentCanvas.getContext('2d');
    const ctrlCtx = ctrlCanvas.getContext('2d');
    if (!sentCtx || !ctrlCtx) return;

    const pct = (n, d) => d ? Math.round((n / d) * 100) : 0;
    const sentimentPct = pct(stats.neg, stats.total);
    const controlPct = pct(stats.ctrl, stats.total);
    const centerTextPlugin = (idSuffix, text) => ({
      id: `centerText-${idSuffix}-${text}`,
      afterDraw(chart) {
        const {ctx} = chart;
        if (!ctx) return;
        const {left, right, top, bottom} = chart.chartArea;
        const x = (left + right) / 2;
        const y = (top + bottom) / 2;
        ctx.save();
        ctx.fillStyle = '#ebf2f2';
        ctx.font = '600 20px "Manrope", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, x, y);
        ctx.restore();
      }
    });

    new Chart(sentCtx, {
      type: 'doughnut',
      data: {
        labels: ['Negative', 'Other'],
        datasets: [{
          data: [stats.neg, Math.max(0, stats.total - stats.neg)],
          backgroundColor: ['#ff826166', 'rgba(207,219,221,0.08)'],
          borderColor: ['#ff8261', '#cfdbdd'],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#ebf2f2' } } }
      },
      plugins: [centerTextPlugin(sentCanvasId, `${sentimentPct}%`)]
    });

    new Chart(ctrlCtx, {
      type: 'doughnut',
      data: {
        labels: ['Controlled', 'Uncontrolled/Unknown'],
        datasets: [{
          data: [stats.ctrl, Math.max(0, stats.total - stats.ctrl)],
          backgroundColor: ['#58dbed66', 'rgba(88,219,237,0.03)'],
          borderColor: ['#58dbed', '#58dbed'],
          borderWidth: [1, 2]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#ebf2f2' } } }
      },
      plugins: [centerTextPlugin(ctrlCanvasId, `${controlPct}%`)]
    });
  }

  function renderExternalSummaryMessage(outputId){
    const output = modalContent.querySelector(`#${outputId}`);
    if (!output) return;
    output.textContent = 'AI summary is available in the internal dashboard.';
  }

  function showBoards(ceo){
    modalEntityState = null;
    const key = canonName(ceo);
    let boards = boardsByCeo.get(ceo) || boardsByCeo.get(key);

    if (!boards){
      const k = [...boardsByCeo.keys()].find(k => k === key);
      boards = k ? boardsByCeo.get(k) : null;
    }

    if (!boards || !boards.length){
      openModal(`Boards — ${ceo}`, `<div class="muted">No boards found.</div>`);
      return;
    }

    const html = `<ul style="padding-left:20px;margin:0">
      ${boards.map(b=>{
        const text = esc(b.domain || (b.url ? new URL(b.url).hostname.replace(/^www\./,'') : 'Board'));
        const href = esc(b.url || '#');
        return `<li style="margin:6px 0"><a class="link" href="${href}" target="_blank" rel="noopener">${text}</a></li>`;
      }).join('')}
    </ul>`;
    openModal(`Boards — ${ceo}`, html);
  }

  async function showHeadlines({ ceo, company }, modalDate = null) {
    const ctx = normalizeCeoModalContext({ ceo, company });
    setCeoModalState(ctx, 'headlines');
    ceo = ctx.ceo;
    const d = String(modalDate || getModalDate()).trim();
    syncModalDateOverride(d);
    const modalOpts = { contextDate: d, onDateChange: (nextDate) => showHeadlines({ ceo, company: ctx.company }, nextDate) };
    openModal(`Headlines — ${esc(ceo)}`, `<div class="muted">Loading…</div>`, modalOpts);
    const cacheKey = `${d}|${ceo}`;
    let rows = [];
    let total = null;
    let offset = 0;
    try {
      if (_modalHeadlinesCache.has(cacheKey)) {
        const cached = _modalHeadlinesCache.get(cacheKey);
        rows = cached.rows || [];
        total = cached.total;
      } else {
        const data = await fetchCsv(HEADLINES_PATH_CEO(d, ceo, offset));
        rows = data.rows || data;
        total = data.total;
        _modalHeadlinesCache.set(cacheKey, { rows, total });
      }
    } catch (e) {
      openModal(`Headlines — ${esc(ceo)}`,
        `<div class="muted">Could not load headlines for ${d}.</div>`, modalOpts);
      return;
    }

    let list = rows.filter(r => String(r.ceo || '').trim() === ceo);

    if (!list.length) {
      openModal(`Headlines — ${esc(ceo)}`,
        `<div class="muted">No headlines for ${esc(ceo)} on ${d}.</div>`, modalOpts);
      return;
    }

    const renderCards = (items) => items.map(r => {
      const url = String(r.url || '').trim();
      let domain = '';
      try { domain = new URL(url).hostname.replace(/^www\./, ''); } catch {}
      const source = esc(r.source || '');
      const sourceLabel = source || esc(domain || '');
      const publishedLabel = esc(articlePublishedMeta(r));
      const metaLabel = [sourceLabel, publishedLabel].filter(Boolean).join(' · ');
      const title = esc(r.title || domain || '(no title)');
      const link = url
        ? `<a class="link" href="${esc(url)}" target="_blank" rel="noopener">${title}</a>`
        : `<span class="muted">${title}</span>`;

      const sRaw = String(r.sentiment || '').toLowerCase();
      const s = sRaw === 'positive' ? 'positive' : (sRaw === 'negative' ? 'negative' : 'neutral');
      const ctrlRaw = String(r.control_override || r.control_class || '').toLowerCase();
      const ctrl = ctrlRaw === 'controlled' ? 'controlled' : (ctrlRaw === 'uncontrolled' ? 'uncontrolled' : '');
      const flags = flagIcons(r);

      return `
        <div class="serp-card">
          ${metaLabel ? `<div class="serp-domain">${metaLabel}</div>` : ''}
          <div class="serp-title">${link}</div>
          <div class="badges" style="margin-top:8px">
            <span class="badge ${s}">${s}</span>
            ${ctrl ? `<span class="badge ${ctrl}">${ctrl}</span>` : ''}
            ${flags ? `<span class="edit-flags">${flags}</span>` : ''}
          </div>
        </div>
      `;
    }).join('');

    const renderModal = () => {
      const cards = renderCards(list);
      const stats = computeModalStats(
        list,
        (r) => (/^controlled|true|1$/i).test(String(r.control_override || r.control_class || r.controlled || ''))
      );
      const negPct = stats.total ? (stats.neg / stats.total) : 0;
      const ctrlPct = stats.total ? (stats.ctrl / stats.total) : 0;
      const showMore = total ? list.length < total : list.length % MODAL_LIMIT === 0;
      const moreBtn = showMore ? `<button id="modalLoadMore" class="ghost-btn">Load more</button>` : '';
      const totalLabel = total ? `<div class="muted" style="margin-bottom:8px">Showing ${list.length} of ${total}</div>` : '';
      const metrics = `<div class="serp-metrics">
        <b>Negative Headlines:</b> ${(negPct*100).toFixed(1)}% &nbsp;|&nbsp;
        <b>Control:</b> ${(ctrlPct*100).toFixed(1)}%
      </div>`;
      const insights = `
        ${metrics}
        <div class="serp-feature-viz" style="margin-top:10px">
          <div style="flex:1 1 220px;height:220px"><canvas id="headlineSentChart"></canvas></div>
          <div style="flex:1 1 220px;height:220px"><canvas id="headlineControlChart"></canvas></div>
        </div>
        <div class="serp-card serp-feature-summary">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
            <h3 style="margin:0">AI Summary</h3>
            <button class="ghost-btn" disabled>Internal only</button>
          </div>
          <div id="headlineSummaryText" class="serp-snippet muted"></div>
        </div>
      `;
      openModal(`Headlines — ${esc(ceo)}`, `${totalLabel}${insights}<div class="serp-cards">${cards}</div>${moreBtn}`, modalOpts);
      renderModalInsightDonuts('headlineSentChart', 'headlineControlChart', stats);
      renderExternalSummaryMessage('headlineSummaryText');
      const btn = modalContent.querySelector('#modalLoadMore');
      if (btn) {
        btn.onclick = async () => {
          btn.disabled = true;
          offset = list.length;
          const data = await fetchCsv(HEADLINES_PATH_CEO(d, ceo, offset));
          const more = data.rows || data;
          if (Array.isArray(more) && more.length) {
            rows = rows.concat(more);
            if (data.total) total = data.total;
            _modalHeadlinesCache.set(cacheKey, { rows, total });
            list = rows.filter(r => String(r.ceo || '').trim() === ceo);
          }
          renderModal();
        };
      }
    };

    renderModal();
  }

  async function showSerp(row, modalDate = null){
    row = normalizeCeoModalContext(row);
    setCeoModalState(row, 'serp');
    const d = String(modalDate || getModalDate()).trim();
    syncModalDateOverride(d);
    const modalOpts = { contextDate: d, onDateChange: (nextDate) => showSerp(row, nextDate) };
    openModal(`SERP — ${esc(row.ceo)}`, `<div class="muted">Loading…</div>`, modalOpts);
    const cacheKey = `${d}|${row.ceo}`;
    let rows = [];
    let total = null;
    try {
      if (_modalSerpCache.has(cacheKey)) {
        const cached = _modalSerpCache.get(cacheKey);
        rows = cached.rows || [];
        total = cached.total;
      } else {
        const data = await fetchCsv(SERP_ROWS_PATH(d, row.ceo, 0));
        rows = data.rows || data;
        total = data.total;
        _modalSerpCache.set(cacheKey, { rows, total });
      }
    }
    catch { openModal(`SERP — ${esc(row.ceo)}`, `<div class="muted">Could not load SERP file for ${d}.</div>`, modalOpts); return; }
    if (!rows.length){ openModal(`SERP — ${esc(row.ceo)}`, `<div class="muted">No SERP data found for ${d}.</div>`, modalOpts); return; }

    const canon = s => String(s||'').toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();

    const wantCEO = canon(row.ceo), wantCo = canon(row.company);
    let matches = rows.filter(r => {
      const rCEO = canon(r.ceo), rCo = canon(r.company);
      if (wantCo) return (r.ceo===row.ceo && r.company===row.company) || (rCEO===wantCEO && rCo===wantCo);
      return (r.ceo===row.ceo) || (rCEO===wantCEO);
    });

    if (!matches.length){
      openModal(`SERP — ${esc(row.ceo)}`, `<div class="muted">No SERPs for ${esc(row.ceo)} on ${d}.</div>`, modalOpts); return;
    }

    const totalMatches = matches.length;
    const negCount = matches.filter(r => String(r.sentiment||'').toLowerCase()==='negative').length;
    const ctrlCount = matches.filter(r => (/^controlled|true|1$/i).test(String(r.controlled||''))).length;
    const negPct = totalMatches ? (negCount/totalMatches) : 0;
    const ctrlPct = totalMatches ? (ctrlCount/totalMatches) : 0;
    const risk = computeRisk(negPct, ctrlPct);

    matches.sort((a,b)=>(+a.position||9999) - (+b.position||9999));

    const renderCards = (items) => items.map(r=>{
      const url = String(r.url||'').trim();
      let domain = '';
      try { domain = new URL(url).hostname.replace(/^www\./,''); } catch {}
      const metaLabel = esc(buildSerpMetaLabel(r, domain));
      const title = esc(r.title || domain || '(no title)');
      const snippet = esc(r.snippet || '');
      const s = String(r.sentiment||'neutral').toLowerCase();
      const ctrl = (/^controlled|true|1$/i).test(String(r.controlled||'')) ? 'controlled' : 'uncontrolled';
      const link = url ? `<a class="link" href="${esc(url)}" target="_blank" rel="noopener">${title}</a>` : `<span class="muted">${title}</span>`;
      const flags = flagIcons(r);
      return `<div class="serp-card">
        ${metaLabel ? `<div class="serp-domain">${metaLabel}</div>` : ''}
        <div class="serp-title">${link}</div>
        ${snippet? `<div class="serp-snippet">${snippet}</div>` : ''}
        <div class="badges">
          <span class="badge ${s}">${s}</span>
          <span class="badge ${ctrl}">${ctrl}</span>
          ${flags ? `<span class="edit-flags">${flags}</span>` : ''}
        </div>
      </div>`;
    }).join('');

    const metrics = `<div class="serp-metrics">
      <b>Negative SERP:</b> ${(negPct*100).toFixed(1)}% &nbsp;|&nbsp;
      <b>Control:</b> ${(ctrlPct*100).toFixed(1)}% &nbsp;|&nbsp;
      <b>Risk:</b> ${esc(risk)}
    </div>`;

    const renderModal = () => {
      const cards = renderCards(matches);
      const stats = computeModalStats(
        matches,
        (r) => (/^controlled|true|1$/i).test(String(r.controlled || r.control_override || r.control_class || ''))
      );
      const showMore = total ? matches.length < total : matches.length % MODAL_LIMIT === 0;
      const moreBtn = showMore ? `<button id="modalLoadMore" class="ghost-btn">Load more</button>` : '';
      const totalLabel = total ? `<div class="muted" style="margin:6px 0 8px">Showing ${matches.length} of ${total}</div>` : '';
      const insights = `
        <div class="serp-feature-viz" style="margin-top:10px">
          <div style="flex:1 1 220px;height:220px"><canvas id="serpSentChart"></canvas></div>
          <div style="flex:1 1 220px;height:220px"><canvas id="serpControlChart"></canvas></div>
        </div>
        <div class="serp-card serp-feature-summary">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
            <h3 style="margin:0">AI Summary</h3>
            <button class="ghost-btn" disabled>Internal only</button>
          </div>
          <div id="serpSummaryText" class="serp-snippet muted"></div>
        </div>
      `;
      openModal(`SERP — ${esc(row.ceo)}`, `${metrics}${totalLabel}${insights}<div class="serp-cards">${cards}</div>${moreBtn}`, modalOpts);
      renderModalInsightDonuts('serpSentChart', 'serpControlChart', stats);
      renderExternalSummaryMessage('serpSummaryText');
      const btn = modalContent.querySelector('#modalLoadMore');
      if (btn) {
        btn.onclick = async () => {
          btn.disabled = true;
          const offset = matches.length;
          const data = await fetchCsv(SERP_ROWS_PATH(d, row.ceo, offset));
          const more = data.rows || data;
          if (Array.isArray(more) && more.length) {
            rows = rows.concat(more);
            if (data.total) total = data.total;
            _modalSerpCache.set(cacheKey, { rows, total });
            matches = rows.filter(r => {
              const rCEO = canon(r.ceo), rCo = canon(r.company);
              if (wantCo) return (r.ceo===row.ceo && r.company===row.company) || (rCEO===wantCEO && rCo===wantCo);
              return (r.ceo===row.ceo) || (rCEO===wantCEO);
            });
          }
          renderModal();
        };
      }
    };

    renderModal();
  }

  async function fetchSerpFeatureItems(date, ceo, feature, offset, limit){
    const featureKey = feature || FEATURE_ALL_KEY;
    const key = `${date}|${ceo}|${featureKey}|${offset}|${limit}`;
    if (_serpFeatureItemsCache.has(key)) return _serpFeatureItemsCache.get(key);
    const featureParam = feature === FEATURE_ALL_KEY ? '' : feature;
    const rows = await fetchCsv(SERP_FEATURE_ITEMS_PATH(date, ceo, featureParam, offset, limit));
    _serpFeatureItemsCache.set(key, rows);
    return rows;
  }

  async function showSerpFeatures({ceo, company}, modalDate = null){
    const ctx = normalizeCeoModalContext({ ceo, company });
    setCeoModalState(ctx, 'features');
    ceo = ctx.ceo;
    const d = String(modalDate || getModalDate()).trim();
    syncModalDateOverride(d);
    const modalOpts = { contextDate: d, onDateChange: (nextDate) => showSerpFeatures({ ceo, company: ctx.company }, nextDate) };
    openModal(`SERP Features — ${esc(ceo)}`, `<div class="muted">Loading…</div>`, modalOpts);
    await loadSerpFeaturesForEntity(ceo);

    const totalsByFeature = new Map();
    serpFeatureEntityRows.forEach(r=>{
      if (r.date !== d || r.entity !== ceo) return;
      if (!isItemFeature(r.feature)) return;
      totalsByFeature.set(r.feature, (totalsByFeature.get(r.feature) || 0) + (r.total || 0));
    });
    const totalAll = FEATURE_MODAL_ORDER.reduce((sum, f) => sum + (totalsByFeature.get(f) || 0), 0);
    const options = [
      `<option value="${FEATURE_ALL_KEY}">${FEATURE_LABELS[FEATURE_ALL_KEY]} (${totalAll})</option>`,
      ...FEATURE_MODAL_ORDER.map(f=>{
        const count = totalsByFeature.get(f) ?? 0;
        const label = FEATURE_LABELS[f] || f;
        return `<option value="${f}">${label} (${count})</option>`;
      })
    ].join('');

    openModal(`SERP Features — ${esc(ceo)}`, `
      <div class="serp-metrics">
        <label for="serpFeatureSelect"><b>Feature</b></label>
        <select id="serpFeatureSelect">${options}</select>
        <span id="serpFeatureTotal" class="muted" style="margin-left:12px"></span>
      </div>
      <div id="serpFeatureCounts" class="muted" style="margin-top:6px"></div>
      <div id="serpFeatureList" class="serp-cards"><div class="muted">Loading…</div></div>
      <div class="pagination" id="serpFeaturePager" style="display:none">
        <button id="serpFeatureMore">Load more</button>
      </div>
    `, modalOpts);

    const select = modalContent.querySelector('#serpFeatureSelect');
    const listEl = modalContent.querySelector('#serpFeatureList');
    const totalEl = modalContent.querySelector('#serpFeatureTotal');
    const countsEl = modalContent.querySelector('#serpFeatureCounts');
    const pager = modalContent.querySelector('#serpFeaturePager');
    const moreBtn = modalContent.querySelector('#serpFeatureMore');
    let currentOffset = 0;

    const featuresFor = (feature) => (
      feature === FEATURE_ALL_KEY ? FEATURE_MODAL_ORDER : [feature]
    );

    const computeCountsForDate = (feature) => {
      const features = featuresFor(feature);
      let total = 0;
      let neg = 0;
      serpFeatureEntityRows.forEach(r=>{
        if (r.date !== d || r.entity !== ceo) return;
        if (!features.includes(r.feature)) return;
        total += r.total || 0;
        neg += r.neg || 0;
      });
      if (!total) return null;
      return { neg, total };
    };

    async function renderFeature(reset = false){
      const feature = select.value;
      const label = FEATURE_LABELS[feature] || feature;
      if (reset){
        currentOffset = 0;
        listEl.innerHTML = '<div class="muted">Loading…</div>';
      }
      const counts = computeCountsForDate(feature);
      totalEl.textContent = counts ? `Total URLs: ${counts.total}` : '';
      countsEl.textContent = counts
        ? `Counts for ${d}: ${counts.neg} negative (total ${counts.total})`
        : `No items for ${d}.`;

      let rows = [];
      try{
        rows = await fetchSerpFeatureItems(d, ceo, feature, currentOffset, FEATURE_MODAL_LIMIT);
      }catch{
        listEl.innerHTML = `<div class="muted">Could not load ${label} items.</div>`;
        pager.style.display = 'none';
        return;
      }
      if (!rows.length && currentOffset === 0){
        listEl.innerHTML = `<div class="muted">No ${label} items for ${esc(ceo)} on ${d}.</div>`;
        pager.style.display = 'none';
        return;
      }

      const cards = rows.map(r=>{
        const url = String(r.url||'').trim();
        let domain = String(r.domain||'').trim();
        if (!domain && url){ try{ domain = new URL(url).hostname.replace(/^www\./,''); }catch{} }
        const title = esc(r.title || domain || '(no title)');
        const snippet = esc(r.snippet || '');
        const metaLabel = esc(buildSerpMetaLabel(r, domain));
        const s = String(r.sentiment||'neutral').toLowerCase();
        const sentiment = s === 'positive' ? 'positive' : (s === 'negative' ? 'negative' : 'neutral');
        const controlClass = String(r.control_class || r.controlled || '').toLowerCase();
        const ctrl = (/^controlled|true|1$/i).test(controlClass) ? 'controlled' : 'uncontrolled';
        const link = url ? `<a class="link" href="${esc(url)}" target="_blank" rel="noopener">${title}</a>` : `<span class="muted">${title}</span>`;
        const featureLabel = FEATURE_LABELS[r.feature_type] || r.feature_type || '';
        const featureFlag = featureLabel ? `<div class="serp-feature-flag">${esc(featureLabel)}</div>` : '';
        const flags = flagIcons(r);
        return `<div class="serp-card">
          ${featureFlag}
          ${metaLabel ? `<div class="serp-domain">${metaLabel}</div>` : ''}
          <div class="serp-title">${link}</div>
          ${snippet ? `<div class="serp-snippet">${snippet}</div>` : ''}
          <div class="badges">
            <span class="badge ${sentiment}">${sentiment}</span>
            <span class="badge ${ctrl}">${ctrl}</span>
            ${flags ? `<span class="edit-flags">${flags}</span>` : ''}
          </div>
        </div>`;
      }).join('');

      if (currentOffset === 0) listEl.innerHTML = cards;
      else listEl.insertAdjacentHTML('beforeend', cards);
      pager.style.display = rows.length === FEATURE_MODAL_LIMIT ? 'flex' : 'none';
    }

    select.onchange = () => { renderFeature(true); };
    moreBtn.onclick = () => {
      currentOffset += FEATURE_MODAL_LIMIT;
      renderFeature(false);
    };
    await renderFeature(true);
  }

  init();
}

export async function prefetchParityTab({ getDirectUrl }) {
  return prefetchParityPage({ page, getDirectUrl });
}

export async function mountParityTab({ host, getDirectUrl, onHistoryChange }) {
  return mountParityPage({
    host,
    getDirectUrl,
    onHistoryChange,
    page,
    runInlineScript,
  });
}
