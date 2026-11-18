#!/usr/bin/env node
/**
 * replace-emojis-svg.mjs
 *
 * Replaces all remaining legacy emoji icons (Design Rule 1) with clean inline
 * stroke SVG icons across HTML markup, and clean architectural text/badges
 * inside CSS pseudo-elements and JS string literals.
 */

import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const HERE = path.dirname(url.fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

const EMOJI_RE = /(?!©)\p{Extended_Pictographic}(\uFE0F|\u20E3)?/gu;

const svg = (inner, size = 15, extraAttr = '') =>
  `<svg class="inline-icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:-0.15em;flex-shrink:0;" aria-hidden="true"${extraAttr}>${inner}</svg>`;

const dotSvg = (color) =>
  `<svg class="inline-icon" width="10" height="10" viewBox="0 0 10 10" style="display:inline-block;vertical-align:0.05em;flex-shrink:0;" aria-hidden="true"><circle cx="5" cy="5" r="4" fill="${color}"/></svg>`;

const ICONS = {
  // Circles / status dots
  '🔴': dotSvg('#ef4444'),
  '🟠': dotSvg('#f97316'),
  '🟡': dotSvg('#eab308'),
  '🟢': dotSvg('#22c55e'),
  '🔵': dotSvg('#3b82f6'),
  '🟣': dotSvg('#a855f7'),
  '🩵': dotSvg('#06b6d4'),
  '🔷': svg('<polygon points="12 2 22 12 12 22 2 12 12 2"/>'),
  '⭕': svg('<circle cx="12" cy="12" r="9"/>'),

  // Checks, crosses, warnings
  '✅': svg('<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'),
  '❌': svg('<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>'),
  '🚫': svg('<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>'),
  '⚠️': svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  '⚠': svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  '🚨': svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),

  // Targets, ideas, books, labs
  '🎯': svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'),
  '💡': svg('<path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/>'),
  '💭': svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
  '🤔': svg('<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  '❓': svg('<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  '📖': svg('<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'),
  '📚': svg('<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>'),
  '📗': svg('<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>'),
  '🎓': svg('<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>'),
  '🔬': svg('<path d="M9 3h6"/><path d="M10 9V3"/><path d="M14 9V3"/><path d="M6 21h12"/><path d="M12 17v4"/><path d="M8 13l4-4 4 4a4 4 0 0 1-8 0z"/>'),
  '🧪': svg('<path d="M9 3h6"/><path d="M10 3v6l-4.5 8.5A2 2 0 0 0 7.2 20h9.6a2 2 0 0 0 1.7-2.5L14 9V3"/>'),
  '🔭': svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/>'),

  // Interactive, AI, Search
  '🎮': svg('<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>'),
  '🎛️': svg('<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'),
  '🎲': svg('<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1"/><circle cx="15.5" cy="15.5" r="1"/><circle cx="12" cy="12" r="1"/>'),
  '🧠': svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>'),
  '🤖': svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="1" y1="9" x2="4" y2="9"/>'),
  '🔍': svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'),
  '🔎': svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'),
  '👁️': svg('<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>'),

  // Places, Docs, Code
  '🏠': svg('<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>'),
  '🏛️': svg('<path d="M3 21h18"/><path d="M5 21V7l7-4 7 4v14"/><path d="M9 21v-8h6v8"/>'),
  '🏢': svg('<rect x="4" y="2" width="16" height="20" rx="2"/><line x1="9" y1="6" x2="9" y2="6.01"/><line x1="15" y1="6" x2="15" y2="6.01"/><line x1="9" y1="10" x2="9" y2="10.01"/><line x1="15" y1="10" x2="15" y2="10.01"/>'),
  '🏭': svg('<path d="M2 20h20"/><path d="M4 20V8l6 4V8l6 4V4h4v16"/>'),
  '📝': svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'),
  '📄': svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>'),
  '📋': svg('<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/>'),
  '📜': svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="14" y2="17"/>'),
  '✍️': svg('<path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>'),
  '🗂️': svg('<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>'),
  '📂': svg('<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>'),
  '📦': svg('<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>'),
  '🗄️': svg('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'),
  '💾': svg('<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/>'),
  '💻': svg('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>'),
  '🖥️': svg('<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>'),
  '🐍': svg('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>'),

  // Maps, Charts, Scales
  '🗺️': svg('<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>'),
  '🧭': svg('<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'),
  '🌍': svg('<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'),
  '🌐': svg('<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'),
  '📌': svg('<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>'),
  '📍': svg('<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>'),
  '✂️': svg('<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/>'),
  '📊': svg('<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'),
  '📈': svg('<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>'),
  '📉': svg('<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>'),
  '🧮': svg('<rect x="4" y="2" width="16" height="20" rx="2"/><line x1="8" y1="6" x2="16" y2="6"/><line x1="8" y1="10" x2="16" y2="10"/>'),
  '🔢': svg('<line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/>'),
  '🔤': svg('<path d="M4 20h16"/><path d="M8 16l4-10 4 10"/><path d="M9.5 12h5"/>'),
  '📐': svg('<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0z"/>'),
  '📏': svg('<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0z"/>'),
  '⚖️': svg('<line x1="12" y1="3" x2="12" y2="21"/><path d="M5 7h14"/><path d="M3 14l2-7 2 7a2 2 0 0 1-4 0z"/><path d="M17 14l2-7 2 7a2 2 0 0 1-4 0z"/>'),
  '⚖': svg('<line x1="12" y1="3" x2="12" y2="21"/><path d="M5 7h14"/><path d="M3 14l2-7 2 7a2 2 0 0 1-4 0z"/><path d="M17 14l2-7 2 7a2 2 0 0 1-4 0z"/>'),

  // Security, Tools, Energy
  '🔑': svg('<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>'),
  '🗝️': svg('<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777z"/>'),
  '🔐': svg('<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'),
  '🔒': svg('<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'),
  '🔓': svg('<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/>'),
  '🛡️': svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
  '🛡': svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
  '⚙️': svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'),
  '⚙': svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'),
  '🔧': svg('<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'),
  '🛠️': svg('<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'),
  '🏗️': svg('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/>'),
  '🧹': svg('<path d="M19 7l-7 7"/><path d="M5 21l4-4"/><path d="M12 14l-5 5-2-2 5-5z"/>'),
  '🗜️': svg('<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>'),
  '⚡': svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'),
  '🔥': svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'),
  '🚀': svg('<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'),
  '⭐': svg('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>'),
  '🏆': svg('<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/>'),

  // Arrows, Links, Flags, Misc
  '▶': svg('<polyline points="9 18 15 12 9 6"/>', 12),
  '🔄': svg('<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'),
  '🔁': svg('<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>'),
  '🔀': svg('<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/>'),
  '⏩': svg('<polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/>'),
  '⏪': svg('<polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/>'),
  '⏭': svg('<polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/>'),
  '⬇️': svg('<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>'),
  '↔️': svg('<polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/>'),
  '↕️': svg('<polyline points="8 6 12 2 16 6"/><polyline points="8 18 12 22 16 18"/><line x1="12" y1="2" x2="12" y2="22"/>'),
  '↗️': svg('<line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/>'),
  '↘️': svg('<line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/>'),
  '↙️': svg('<line x1="17" y1="7" x2="7" y2="17"/><polyline points="17 17 7 17 7 7"/>'),
  '↖️': svg('<line x1="17" y1="17" x2="7" y2="7"/><polyline points="7 17 7 7 17 7"/>'),
  '🔗': svg('<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'),
  '📎': svg('<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>'),
  '🕸️': svg('<circle cx="12" cy="12" r="9"/><line x1="12" y1="3" x2="12" y2="21"/><line x1="3" y1="12" x2="21" y2="12"/>'),
  '🧩': svg('<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'),
  '🧲': svg('<path d="M6 15V9a6 6 0 0 1 12 0v6"/><line x1="6" y1="15" x2="9" y2="15"/><line x1="15" y1="15" x2="18" y2="15"/>'),
  '🔌': svg('<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8z"/>'),
  '🚩': svg('<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>'),
  '🏁': svg('<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>'),
  '📡': svg('<path d="M5.6 5.6a9 9 0 0 1 12.8 0"/><path d="M8.5 8.5a5 5 0 0 1 7 0"/><circle cx="12" cy="12" r="1"/><line x1="12" y1="13" x2="12" y2="21"/>'),
  '🏷️': svg('<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>'),
  '💰': svg('<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>'),
  '🌳': svg('<polygon points="12 2 4 14 20 14 12 2"/><line x1="12" y1="14" x2="12" y2="22"/>'),
  '🌓': svg('<circle cx="12" cy="12" r="10"/><path d="M12 2v20"/>'),
  '➕': svg('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>'),
  '📥': svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>'),
  '📤': svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'),
  '🔇': svg('<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>'),
  '🔔': svg('<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>'),
  '📧': svg('<rect x="2" y="4" width="20" height="16" rx="2"/><polyline points="22,6 12,13 2,6"/>'),
  '💬': svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
  '🗣️': svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
  '🤝': svg('<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'),
  '🛍️': svg('<path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/><line x1="3" y1="6" x2="21" y2="6"/>'),
  '🛒': svg('<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>'),
  '⚕️': svg('<path d="M12 2v20"/><path d="M5 12h14"/>'),
  '🐛': svg('<rect x="8" y="6" width="8" height="14" rx="4"/><path d="M19 10h2"/><path d="M3 10h2"/><path d="M19 14h2"/><path d="M3 14h2"/>'),
  '🌊': svg('<path d="M2 12c2-2 4-2 6 0s4 2 6 0 4-2 6 0"/><path d="M2 17c2-2 4-2 6 0s4 2 6 0 4-2 6 0"/>'),
  '🎭': svg('<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>'),
  '🔮': svg('<circle cx="12" cy="12" r="9"/><path d="M12 3v18"/>'),
  '🏹': svg('<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>'),
  '🏊': svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
  '😤': svg('<circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/>'),
  '😰': svg('<circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/>'),
  '🩷': dotSvg('#ec4899'),
  '🦦': svg('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>'),
};

const FALLBACK_SVG = svg('<polygon points="12 2 15 9 22 12 15 15 12 22 9 15 2 12 9 9 12 2"/>');

const SCRIPT_TEXT_MAP = {
  '✅': '[OK]',
  '❌': '[X]',
  '⚠️': '[WARN]',
  '⚠': '[WARN]',
  '🚨': '[ALERT]',
  '💥': '[EXPLODING]',
  '🌫️': '[VANISHING]',
  '🌫': '[VANISHING]',
  '🟡': '[LOW]',
  '🟢': '[MEDIUM]',
  '🔵': '[HIGH]',
  '🎯': '',
  '🔄': '',
  '🚩': '[FLAG]',
  '🎉': '',
  '📚': '',
  '💪': '',
};

const targets = [
  'aws_ai_practitioner_exam.html',
  'DP-700-InteractiveStudy.html',
  ...fs.readdirSync(path.join(ROOT, 'GaTech/CSE 8803')).filter((f) => f.endsWith('.html')).map((f) => `GaTech/CSE 8803/${f}`),
  ...fs.readdirSync(path.join(ROOT, 'aip')).filter((f) => f.endsWith('.html')).map((f) => `aip/${f}`),
];

function replaceHtmlChunk(chunk) {
  // Replace HTML attribute placeholders or titles containing emojis first
  let out = chunk.replace(/\b(placeholder|title|aria-label)="([^"]*)"/g, (full, attr, val) => {
    if (!EMOJI_RE.test(val)) return full;
    const cleaned = val.replace(EMOJI_RE, '').replace(/^\s+/, '');
    return `${attr}="${cleaned}"`;
  });

  // Replace remaining emojis in HTML text with inline SVGs
  out = out.replace(EMOJI_RE, (m) => {
    const icon = ICONS[m] || ICONS[m.replace(/\uFE0F|\u20E3/g, '')] || FALLBACK_SVG;
    return icon;
  });
  return out;
}

function processFile(rel) {
  const abs = path.join(ROOT, rel);
  let txt = fs.readFileSync(abs, 'utf8');
  const orig = txt;

  // 1. Special context: aip/study-d1..d45.html CSS pseudo-elements
  if (/^aip\/study-d\d+\.html$/.test(rel)) {
    txt = txt
      .replace(/\.tip::before\{content:"💡 "\}/g, '.tip::before{content:"TIP · ";font-weight:800;letter-spacing:.04em;}')
      .replace(/\.gotcha::before\{content:"⚠️ "\}/g, '.gotcha::before{content:"TRAP · ";font-weight:800;letter-spacing:.04em;}')
      .replace(/\.remember::before\{content:"🧠 "\}/g, '.remember::before{content:"KEY · ";font-weight:800;letter-spacing:.04em;}')
      .replace(/\.when::before\{content:"🎯 "\}/g, '.when::before{content:"USE · ";font-weight:800;letter-spacing:.04em;}')
      .replace(/\.avoid::before\{content:"🚫 "\}/g, '.avoid::before{content:"AVOID · ";font-weight:800;letter-spacing:.04em;}')
      .replace(/\.note::before\{content:"📎 "\}/g, '.note::before{content:"NOTE · ";font-weight:800;letter-spacing:.04em;}');
  }

  // 2. Special context: aip/gap-cheatsheet.html CSS pseudo-elements
  if (rel === 'aip/gap-cheatsheet.html') {
    txt = txt
      .replace(/\.note::before\s*\{\s*content:\s*"💡 ";\s*\}/g, '.note::before { content: "TIP · "; font-weight: 800; letter-spacing: .04em; }')
      .replace(/\.rule::before\s*\{\s*content:\s*"📌 ";\s*\}/g, '.rule::before { content: "RULE · "; font-weight: 800; letter-spacing: .04em; }')
      .replace(/\.warn::before\s*\{\s*content:\s*"⚠️ ";\s*\}/g, '.warn::before { content: "TRAP · "; font-weight: 800; letter-spacing: .04em; }');
  }

  // 3. Special context: GaTech/CSE 8803/week1.html NLP sample strings
  if (rel === 'GaTech/CSE 8803/week1.html') {
    txt = txt
      .replace(/Check out https:\/\/abc\.com 😊/g, 'Check out https://abc.com :)')
      .replace(/Score: 100% 😊/g, 'Score: 100% :)');
  }

  // 4. Special context: aws_ai_practitioner_exam.html React JSX exam tips
  if (rel === 'aws_ai_practitioner_exam.html') {
    txt = txt
      .replace(
        '<span className="text-lg select-none">📋</span>',
        '<svg className="w-5 h-5 text-amber-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>'
      )
      .replace('{ icon: "⏰",', '{ icon: "T-30",')
      .replace('{ icon: "🪪",', '{ icon: "ID",')
      .replace('{ icon: "📸",', '{ icon: "DESK",')
      .replace('{ icon: "⌚",', '{ icon: "WATCH",')
      .replace('{ icon: "📶",', '{ icon: "WIFI",')
      .replace('{ icon: "🤫",', '{ icon: "ROOM",')
      .replace('{ icon: "🚩",', '{ icon: "FLAG",')
      .replace('{ icon: "⏱️",', '{ icon: "90M",')
      .replace('{ icon: "🎯",', '{ icon: "PASS",')
      .replace('{ icon: "✅",', '{ icon: "NOTE",')
      .replace(
        '<span className="text-base leading-relaxed flex-shrink-0 mt-0.5 select-none">{icon}</span>',
        '<span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono font-bold text-amber-400 flex-shrink-0 mt-0.5 min-w-[44px]">{icon}</span>'
      );
  }

  // 5. Split by <script>...</script> and <style>...</style> so JS and CSS are handled safely
  const blockRe = /(<script\b[^>]*>[\s\S]*?<\/script>|<style\b[^>]*>[\s\S]*?<\/style>)/gi;
  const parts = txt.split(blockRe);

  const rebuilt = parts.map((part) => {
    if (/^<style\b/i.test(part)) {
      return part;
    }
    if (/^<script\b/i.test(part)) {
      if (!EMOJI_RE.test(part)) return part;
      let s = part;
      // Specific innerHTML templates
      s = s.replace(
        'actions += `<button class="btn btn-outline" onclick="toggleExp()">💡 Explanation</button>`;',
        'actions += `<button class="btn btn-outline" onclick="toggleExp()">Explanation</button>`;'
      );
      s = s.replace('📊 Total:', 'Total:');
      s = s.replace("'✅ Done'", "'✓ Done'");
      s = s.replace("'✅'", "'✓'");
      s = s.replace(/🎯 Quiz Complete!/g, 'Quiz Complete!');
      s = s.replace(/🔄 Restart Quiz/g, 'Restart Quiz');
      s = s.replace(/'🚩 Flagged'/g, "'Flagged'");
      s = s.replace(/'🚩 Flag'/g, "'Flag'");
      s = s.replace(/🎉 Passing Score!/g, 'Passing Score!');
      s = s.replace(/📚 Almost there/g, 'Almost there');
      s = s.replace(/💪 More practice needed/g, 'More practice needed');
      s = s.replace(EMOJI_RE, (m) => {
        const mapped = SCRIPT_TEXT_MAP[m] ?? SCRIPT_TEXT_MAP[m.replace(/\uFE0F|\u20E3/g, '')];
        return mapped !== undefined ? mapped : '';
      });
      return s;
    }
    return replaceHtmlChunk(part);
  }).join('');

  if (rebuilt !== orig) {
    fs.writeFileSync(abs, rebuilt);
    console.log(`  + ${rel}`);
  }
}

for (const rel of targets) {
  processFile(rel);
}
