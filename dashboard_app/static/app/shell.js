import { mountLegacyTab, prefetchLegacyTab } from './legacy-tab-adapter.js';
import { isNativeTab, mountNativeTab, prefetchNativeTab } from './tabs/index.js';

const config = window.__CRISIS_DASHBOARD_CONFIG__ || {};
const tabs = Array.isArray(config.tabs) ? config.tabs : [];
const tabsById = new Map(tabs.map((tab) => [tab.id, tab]));
const nativeParityTabs = new Set(Array.isArray(config.nativeParityTabs) ? config.nativeParityTabs : []);
const tabButtons = new Map(
  Array.from(document.querySelectorAll('.shell-tab[data-tab]')).map((node) => [node.dataset.tab, node]),
);
const host = document.getElementById('tabHost');
const statusNode = document.getElementById('tabStatus');
const openLink = document.getElementById('openTabLink');
const switchDashboardModeLink = document.getElementById('switchDashboardModeLink');

const state = {
  activeTab: null,
  activeDirectHref: '',
  loadToken: 0,
  entries: new Map(),
  prefetches: new Map(),
};

const scheduleIdle = window.requestIdleCallback
  ? (callback) => window.requestIdleCallback(callback)
  : (callback) => window.setTimeout(callback, 0);

function getShellUrl() {
  return new URL(window.location.href);
}

function normalizeTab(tabId) {
  if (tabsById.has(tabId)) return tabId;
  return tabs[0]?.id || '';
}

function shouldUseNativeTab(tabId) {
  if (!isNativeTab(tabId)) return false;
  if (!config.forceLegacyTabs) return true;
  return nativeParityTabs.has(tabId);
}

function getTab(tabId) {
  return tabsById.get(normalizeTab(tabId)) || null;
}

function getDirectTabUrlFromShell(tabId) {
  const tab = getTab(tabId);
  if (!tab) return new URL(window.location.href);
  const url = new URL(tab.path, window.location.origin);
  const shellUrl = getShellUrl();
  const params = new URLSearchParams(shellUrl.search);
  params.delete('tab');
  url.search = params.toString();
  url.hash = shellUrl.hash;
  return url;
}

function getOpenTabUrlFromShell(tabId) {
  const tab = getTab(tabId);
  if (!tab) return new URL(window.location.href);
  const basePath = tab.openPath || tab.path;
  const url = new URL(basePath, window.location.origin);
  const shellUrl = getShellUrl();
  const params = new URLSearchParams(shellUrl.search);
  params.delete('tab');
  url.search = params.toString();
  url.hash = shellUrl.hash;
  return url;
}

function getDirectTabUrl(tabId) {
  const entry = state.entries.get(tabId);
  if (entry?.directHref) {
    return new URL(entry.directHref, window.location.origin);
  }
  return getDirectTabUrlFromShell(tabId);
}

function getOpenTabUrl(tabId) {
  const openUrl = getOpenTabUrlFromShell(tabId);
  const directUrl = getDirectTabUrl(tabId);
  openUrl.search = directUrl.search;
  openUrl.hash = directUrl.hash;
  return openUrl;
}

function rememberDirectUrl(tabId, nextUrl) {
  const entry = getTabEntry(tabId);
  entry.directHref = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
}

function getShellUrlForTab(tabId) {
  const shellUrl = getShellUrl();
  shellUrl.searchParams.set('tab', normalizeTab(tabId));
  return shellUrl;
}

function directToShellUrl(tabId, nextUrl) {
  const normalizedTabId = normalizeTab(tabId);
  const base = getDirectTabUrl(normalizedTabId);
  const resolved = new URL(nextUrl || base.href, base.href);
  const shellUrl = getShellUrlForTab(normalizedTabId);
  shellUrl.search = resolved.search;
  shellUrl.hash = resolved.hash;
  shellUrl.searchParams.set('tab', normalizedTabId);
  return shellUrl;
}

function setStatus(message, stateName = '') {
  if (!statusNode) return;
  statusNode.textContent = message || '';
  if (stateName) statusNode.dataset.state = stateName;
  else delete statusNode.dataset.state;
}

function updateNav(tabId) {
  tabButtons.forEach((button, id) => {
    button.setAttribute('aria-selected', id === tabId ? 'true' : 'false');
  });
}

function updateOpenLink(tabId) {
  if (!openLink || !tabId) return;
  const openTabUrl = getOpenTabUrl(tabId);
  openLink.href = `${openTabUrl.pathname}${openTabUrl.search}${openTabUrl.hash}`;
  openLink.textContent = `Open ${tabsById.get(tabId)?.label || 'tab'} in a new tab`;
}

function getDashboardModeSwitch() {
  const shellUrl = getShellUrl();
  const path = shellUrl.pathname;
  let nextPath = '';
  let label = '';

  if (path === '/' || path === '') {
    nextPath = '/crisis-dashboard/';
    label = 'Switch to Full Dashboard';
  } else if (/\/crisis-dashboard-overview(?:\/|$)/.test(path)) {
    nextPath = path.replace('/crisis-dashboard-overview', '/crisis-dashboard');
    label = 'Switch to Full Dashboard';
  } else if (/\/crisis-dashboard(?:\/|$)/.test(path)) {
    nextPath = path.replace('/crisis-dashboard', '/crisis-dashboard-overview');
    label = 'Switch to Overview Dashboard';
  } else {
    return null;
  }

  const nextUrl = new URL(shellUrl.href);
  nextUrl.pathname = nextPath;
  return {
    href: `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`,
    label,
  };
}

function updateDashboardModeLink() {
  if (!switchDashboardModeLink) return;
  const modeSwitch = getDashboardModeSwitch();
  if (!modeSwitch) {
    switchDashboardModeLink.hidden = true;
    switchDashboardModeLink.removeAttribute('href');
    return;
  }
  switchDashboardModeLink.hidden = false;
  switchDashboardModeLink.href = modeSwitch.href;
  switchDashboardModeLink.textContent = modeSwitch.label;
}

function commitShellUrl(nextUrl, { replace = true } = {}) {
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  const next = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
  if (current === next) return;
  if (replace) {
    window.history.replaceState({}, '', next);
  } else {
    window.history.pushState({}, '', next);
  }
}

function getTabEntry(tabId) {
  if (state.entries.has(tabId)) return state.entries.get(tabId);

  const panel = document.createElement('div');
  panel.className = 'shell-panel';
  panel.hidden = true;
  host.appendChild(panel);

  const entry = {
    panel,
    instance: null,
    directHref: '',
  };
  state.entries.set(tabId, entry);
  return entry;
}

function setActivePanel(tabId) {
  state.entries.forEach((entry, id) => {
    const isActive = id === tabId;
    entry.panel.hidden = !isActive;
    if (entry.instance && typeof entry.instance[isActive ? 'show' : 'hide'] === 'function') {
      entry.instance[isActive ? 'show' : 'hide']();
    }
  });
}

function prefetchTab(tabId, options = {}) {
  if (shouldUseNativeTab(tabId)) {
    return prefetchNativeTab({
      tabId,
      shellConfig: config,
      getDirectUrl: () => getDirectTabUrlFromShell(tabId),
    });
  }
  return prefetchLegacyTab(getDirectTabUrlFromShell(tabId), {
    ...options,
    shellConfig: config,
  });
}

function mountTab(tabId, hostNode, getDirectUrl, onHistoryChange) {
  if (shouldUseNativeTab(tabId)) {
    return mountNativeTab({
      host: hostNode,
      getDirectUrl,
      onHistoryChange,
      tabId,
      shellConfig: config,
    });
  }
  return mountLegacyTab({
    host: hostNode,
    getDirectUrl,
    onHistoryChange,
    shellConfig: config,
  });
}

function queuePrefetch(tabId, { preloadScripts = false, prewarmApis = false } = {}) {
  const nextTab = normalizeTab(tabId);
  if (!nextTab) return Promise.resolve();

  const cacheKey = `${nextTab}:${preloadScripts ? 'scripts' : 'markup'}:${prewarmApis ? 'prewarm' : 'cold'}`;
  if (state.prefetches.has(cacheKey)) return state.prefetches.get(cacheKey);

  const promise = prefetchTab(nextTab, { preloadScripts, prewarmApis })
    .catch((error) => {
      state.prefetches.delete(cacheKey);
      console.warn(`Prefetch failed for ${nextTab}`, error);
    });

  state.prefetches.set(cacheKey, promise);
  return promise;
}

function warmInactiveTabs() {
  const inactiveTabs = tabs.map((tab) => tab.id).filter((tabId) => tabId !== state.activeTab);
  scheduleIdle(() => {
    inactiveTabs.forEach((tabId, index) => {
      window.setTimeout(() => {
        const useLegacy = !shouldUseNativeTab(tabId);
        queuePrefetch(tabId, {
          preloadScripts: useLegacy && !!config.forceLegacyTabs,
          prewarmApis: useLegacy && !!config.forceLegacyTabs,
        });
      }, index * 60);
    });
  });
}

async function activateTab(tabId, { syncUrl = true, historyMode = 'replace', forceRemount = false, useLocationState = false } = {}) {
  const nextTab = normalizeTab(tabId);
  if (!nextTab) {
    setStatus('No dashboard tabs are configured.', 'error');
    return;
  }

  const entry = getTabEntry(nextTab);
  const directUrl = useLocationState ? getDirectTabUrlFromShell(nextTab) : getDirectTabUrl(nextTab);
  const directHref = `${directUrl.pathname}${directUrl.search}${directUrl.hash}`;
  rememberDirectUrl(nextTab, directUrl);
  if (syncUrl) {
    commitShellUrl(directToShellUrl(nextTab, directUrl), { replace: historyMode !== 'push' });
  }

  if (entry.instance && entry.directHref === directHref && !forceRemount) {
    state.activeTab = nextTab;
    state.activeDirectHref = directHref;
    updateNav(nextTab);
    updateOpenLink(nextTab);
    updateDashboardModeLink();
    setActivePanel(nextTab);
    entry.instance.show?.();
    setStatus('');
    warmInactiveTabs();
    return;
  }

  const loadToken = ++state.loadToken;
  state.activeTab = nextTab;
  state.activeDirectHref = directHref;
  updateNav(nextTab);
  updateOpenLink(nextTab);
  updateDashboardModeLink();
  setActivePanel(nextTab);
  setStatus(`Loading ${tabsById.get(nextTab)?.label || nextTab}…`);

  if (entry.instance && (entry.directHref !== directHref || forceRemount)) {
    entry.instance.destroy();
    entry.instance = null;
  }

  try {
    if (!entry.instance) {
      const useLegacy = !shouldUseNativeTab(nextTab);
      entry.panel.hidden = false;
      entry.directHref = directHref;
      await queuePrefetch(nextTab, {
        preloadScripts: useLegacy && !!config.forceLegacyTabs,
        prewarmApis: useLegacy && !!config.forceLegacyTabs,
      });
      entry.instance = await mountTab(
        nextTab,
        entry.panel,
        () => getDirectTabUrl(nextTab),
        (next, options = {}) => {
          const shellUrl = directToShellUrl(nextTab, next);
          const resolved = new URL(next || shellUrl.href, window.location.origin);
          rememberDirectUrl(nextTab, resolved);
          commitShellUrl(shellUrl, options);
          state.activeDirectHref = `${resolved.pathname}${resolved.search}${resolved.hash}`;
          updateOpenLink(nextTab);
          updateDashboardModeLink();
        },
      );
    } else {
      entry.directHref = directHref;
      entry.instance.show?.();
    }

    if (loadToken !== state.loadToken) {
      if (entry.instance) entry.instance.hide();
      return;
    }

    setActivePanel(nextTab);
    setStatus('');
    warmInactiveTabs();
  } catch (error) {
    console.error(error);
    entry.instance = null;
    entry.panel.hidden = false;
    entry.panel.innerHTML = `
      <div class="shell-empty">
        <div>
          <h2>Unable to load this tab.</h2>
          <p>${error?.message || 'An unexpected error occurred while mounting the dashboard tab.'}</p>
        </div>
      </div>
    `;
    setStatus('The selected dashboard tab failed to load.', 'error');
  }
}

function syncFromLocation() {
  const shellUrl = getShellUrl();
  const nextTab = normalizeTab(shellUrl.searchParams.get('tab') || tabs[0]?.id);
  const directUrl = getDirectTabUrlFromShell(nextTab);
  const directHref = `${directUrl.pathname}${directUrl.search}${directUrl.hash}`;
  const entry = getTabEntry(nextTab);
  const requiresRemount = !!(entry.instance && entry.directHref && entry.directHref !== directHref);
  rememberDirectUrl(nextTab, directUrl);
  activateTab(nextTab, { syncUrl: false, forceRemount: requiresRemount, useLocationState: true });
}

tabButtons.forEach((button, tabId) => {
  button.addEventListener('click', () => {
    activateTab(tabId, { historyMode: 'push' });
  });
  button.addEventListener('pointerenter', () => {
    const useLegacy = !shouldUseNativeTab(tabId);
    queuePrefetch(tabId, {
      preloadScripts: useLegacy,
      prewarmApis: useLegacy && !!config.forceLegacyTabs,
    });
  });
  button.addEventListener('focus', () => {
    const useLegacy = !shouldUseNativeTab(tabId);
    queuePrefetch(tabId, {
      preloadScripts: useLegacy,
      prewarmApis: useLegacy && !!config.forceLegacyTabs,
    });
  });
});

window.addEventListener('popstate', () => {
  syncFromLocation();
});

syncFromLocation();
