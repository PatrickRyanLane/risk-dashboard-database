export const LOOKBACK_OPTIONS = [30, 45, 60, 90, 180, 365];

const DEFAULT_SIGNAL_SETTINGS = Object.freeze({
  weights: Object.freeze({
    newsNegative: 0.24,
    organicNegative: 0.24,
    topStoriesNegative: 0.16,
    aioCitationsNegative: 0.12,
    paaNegative: 0.1,
    videosNegative: 0.07,
    perspectivesNegative: 0.07,
    serpControl: 0.1,
  }),
  calibration: Object.freeze({
    objective: 'downside_stock',
    autoIterations: 1200,
    controlWeightMax: 0.5,
  }),
});

const ENTITY_CONFIGS = {
  brands: {
    tabId: 'brands',
    entityType: 'brand',
    label: 'Brands',
    title: 'Brand Monitor',
    description: 'Native crisis monitoring for brand-level news and search pressure.',
    primaryKey: 'company',
    secondaryKey: '',
    defaultDays: 30,
    controlThreshold: 0.4,
    dailyCountsKind: 'brand_articles',
    dailySerpKind: 'brand_serps',
    processedEntity: 'brand',
    legacyFilterParam: 'company',
    signalSettings: DEFAULT_SIGNAL_SETTINGS,
    tableColumns: [
      { key: 'entity', label: 'Brand' },
      { key: 'negNews', label: 'News' },
      { key: 'negSerp', label: 'Organic' },
      { key: 'negTopStories', label: 'Top Stories' },
      { key: 'negAio', label: 'AIO' },
      { key: 'negPaa', label: 'PAA' },
      { key: 'negVideos', label: 'Videos' },
      { key: 'negPerspectives', label: 'Perspectives' },
      { key: 'control', label: 'Control' },
      { key: 'stock', label: 'Stock' },
      { key: 'riskScore', label: 'Signal' },
    ],
  },
  ceos: {
    tabId: 'ceos',
    entityType: 'ceo',
    label: 'CEOs',
    title: 'CEO Monitor',
    description: 'Native crisis monitoring for executive-level news and search pressure.',
    primaryKey: 'ceo',
    secondaryKey: 'company',
    defaultDays: 30,
    controlThreshold: 0.35,
    dailyCountsKind: 'ceo_articles',
    dailySerpKind: 'ceo_serps',
    processedEntity: 'ceo',
    legacyFilterParam: 'company',
    signalSettings: DEFAULT_SIGNAL_SETTINGS,
    tableColumns: [
      { key: 'entity', label: 'CEO' },
      { key: 'company', label: 'Company' },
      { key: 'negNews', label: 'News' },
      { key: 'negSerp', label: 'Organic' },
      { key: 'negTopStories', label: 'Top Stories' },
      { key: 'negAio', label: 'AIO' },
      { key: 'negPaa', label: 'PAA' },
      { key: 'negVideos', label: 'Videos' },
      { key: 'negPerspectives', label: 'Perspectives' },
      { key: 'control', label: 'Control' },
      { key: 'stock', label: 'Stock' },
      { key: 'riskScore', label: 'Signal' },
    ],
  },
};

export function isNativeEntityTab(tabId) {
  return tabId === 'brands' || tabId === 'ceos';
}

export function getEntityConfig(tabId, shellConfig = {}) {
  const base = ENTITY_CONFIGS[tabId];
  if (!base) {
    throw new Error(`Unsupported entity tab: ${tabId}`);
  }
  const tabPath = Array.isArray(shellConfig.tabs)
    ? shellConfig.tabs.find((tab) => tab.id === tabId)?.path || ''
    : '';
  return {
    ...base,
    lookbackOptions: LOOKBACK_OPTIONS,
    view: shellConfig.view || 'external',
    tabPath,
    isInternal: shellConfig.view === 'internal',
    signalSettings: {
      weights: { ...(base.signalSettings?.weights || DEFAULT_SIGNAL_SETTINGS.weights) },
      calibration: { ...(base.signalSettings?.calibration || DEFAULT_SIGNAL_SETTINGS.calibration) },
    },
  };
}
