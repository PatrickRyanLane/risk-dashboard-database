import { clearSharedFetchCache, sharedFetch } from './shared-fetch.js';

const CRISIS_MIN_NEG = 4;
const parsedJsonCache = new Map();
const latestDateCache = new Map();
const storeRegistry = new Map();

export const FEATURE_ORDER_SENTIMENT = [
  'organic',
  'aio_citations',
  'paa_items',
  'videos_items',
  'perspectives_items',
  'top_stories_items',
];

export const FEATURE_LABELS = {
  organic: 'Organic',
  aio_citations: 'AIO citations',
  paa_items: 'PAA',
  videos_items: 'Videos',
  perspectives_items: 'Perspectives',
  top_stories_items: 'Top stories',
};

function normalizeUrl(url) {
  return new URL(url, window.location.origin).href;
}

function buildParsedCacheKey(url, init = {}) {
  const method = String(init.method || 'GET').toUpperCase();
  if (method !== 'GET') return '';
  return normalizeUrl(url);
}

async function fetchJson(url, init = {}) {
  const cacheKey = buildParsedCacheKey(url, init);
  if (cacheKey && parsedJsonCache.has(cacheKey)) {
    const cached = parsedJsonCache.get(cacheKey);
    return cached instanceof Promise ? await cached : cached;
  }

  const pending = (async () => {
    const response = await sharedFetch(url, { cache: 'default', credentials: 'same-origin', ...init });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const payload = await response.json();
        if (payload?.error) detail = payload.error;
      } catch (_error) {
        // Ignore non-JSON errors.
      }
      throw new Error(detail);
    }
    return response.json();
  })();

  if (cacheKey) parsedJsonCache.set(cacheKey, pending);

  try {
    const data = await pending;
    if (cacheKey) parsedJsonCache.set(cacheKey, data);
    return data;
  } catch (error) {
    if (cacheKey) parsedJsonCache.delete(cacheKey);
    throw error;
  }
}

async function fetchLatestDatedJson({ key, buildUrl, maxDays = 7 }) {
  if (latestDateCache.has(key)) {
    const cachedDate = latestDateCache.get(key);
    const rows = await fetchJson(buildUrl(cachedDate));
    return { rows, date: cachedDate, daysBack: 0 };
  }

  const now = new Date();
  const todayUtc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  let lastError = null;

  for (let daysBack = 0; daysBack < maxDays; daysBack += 1) {
    const checkDate = new Date(todayUtc);
    checkDate.setUTCDate(checkDate.getUTCDate() - daysBack);
    const dateStr = checkDate.toISOString().slice(0, 10);
    try {
      const rows = await fetchJson(buildUrl(dateStr));
      latestDateCache.set(key, dateStr);
      return { rows, date: dateStr, daysBack };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error(`No data found for ${key}`);
}

function metricNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function maybeNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function isISODate(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(value || '').trim());
}

function canonBrand(value) {
  return String(value || '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function featureIndexKey(date, brandKey) {
  return `${date}|${brandKey}`;
}

function average(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function safeRatio(numerator, denominator) {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator <= 0) return null;
  return numerator / denominator;
}

function collectDates(...collections) {
  const dates = new Set();
  collections.flat().forEach((row) => {
    const date = String(row?.date || '').trim();
    if (date && isISODate(date)) dates.add(date);
  });
  return Array.from(dates).sort((left, right) => left.localeCompare(right));
}

function buildNestedIndex(rows, brandField, valueFactory, accumulator) {
  const outer = new Map();
  rows.forEach((row) => {
    const date = String(row.date || '').trim();
    const brandKey = canonBrand(row[brandField] || '');
    if (!date || !brandKey) return;
    if (!outer.has(date)) outer.set(date, new Map());
    const inner = outer.get(date);
    const current = inner.get(brandKey) || valueFactory();
    accumulator(current, row);
    inner.set(brandKey, current);
  });
  return outer;
}

function parseCountsRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      date: String(row.date || '').trim(),
      company: String(row.company || row.brand || '').trim(),
      positive: metricNumber(row.positive ?? row.pos),
      neutral: metricNumber(row.neutral ?? row.neu),
      negative: metricNumber(row.negative ?? row.neg),
    }))
    .filter((row) => row.date && row.company);
}

function parseSerpRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      date: String(row.date || '').trim(),
      company: String(row.company || row.brand || '').trim(),
      total: metricNumber(row.total),
      negativeSerp: metricNumber(row.negative_serp ?? row.neg_serp),
      controlled: metricNumber(row.controlled ?? row.control),
    }))
    .filter((row) => row.date && row.company);
}

function parseFeatureRows(rows, controlField = 'negative_count') {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      date: String(row.date || '').trim(),
      entity: String(row.entity_name || row.entity || '').trim(),
      feature: String(row.feature_type || '').trim(),
      total: metricNumber(row.total_count ?? row.total),
      value: metricNumber(row[controlField] ?? row.negative ?? row.controlled),
    }))
    .filter((row) => row.date && row.entity && row.feature);
}

function parseRoster(rows) {
  const sectorFromBrand = new Map();
  const brandsBySector = new Map();
  const fortuneFlags = new Map();

  (Array.isArray(rows) ? rows : []).forEach((row) => {
    const brandRaw = row.company || row.Company || row.brand || row.Brand || '';
    const sector = String(row.sector || row.Sector || '').trim();
    const brandKey = canonBrand(brandRaw);
    if (!brandKey || !sector) return;

    const f500 = String(row['Fortune 500'] || row.fortune_500 || row.fortune500 || '').trim().toLowerCase();
    const f1000 = String(row['Fortune 1000'] || row.fortune_1000 || row.fortune1000 || '').trim().toLowerCase();
    const forbes = String(
      row.Forbes ||
      row['Forbes 100'] ||
      row['Forbes 2000'] ||
      row.forbes ||
      row.forbes_100 ||
      row.forbes_2000 ||
      ''
    ).trim().toLowerCase();

    sectorFromBrand.set(brandKey, sector);
    if (!brandsBySector.has(sector)) brandsBySector.set(sector, new Set());
    brandsBySector.get(sector).add(brandKey);
    fortuneFlags.set(brandKey, {
      f500: ['true', '1', 'yes', 'y', 'x'].includes(f500),
      f1000: ['true', '1', 'yes', 'y', 'x'].includes(f1000),
      forbes: ['true', '1', 'yes', 'y', 'x'].includes(forbes),
    });
  });

  return {
    sectorFromBrand,
    brandsBySector,
    fortuneFlags,
    sectors: Array.from(brandsBySector.keys()).sort((left, right) => left.localeCompare(right)),
  };
}

function parseStockRows(rows) {
  const stockByBrand = new Map();
  (Array.isArray(rows) ? rows : []).forEach((row) => {
    const company = String(row.company || '').trim();
    const brandKey = canonBrand(company);
    if (!brandKey) return;
    stockByBrand.set(brandKey, {
      company,
      ticker: String(row.ticker || '').trim(),
      dailyChange: maybeNumber(row.daily_change_pct ?? row.daily_change ?? row.dailyChange),
      sevenDayChange: maybeNumber(row.seven_day_change_pct ?? row.sevenDayChange),
    });
  });
  return stockByBrand;
}

function buildFeatureIndexes(featureRows, featureControlRows) {
  const featureNegByDateBrand = new Map();
  const featureCtrlByDateBrand = new Map();
  const topStoriesNegByBrandDay = new Map();

  featureRows.forEach((row) => {
    if (!FEATURE_ORDER_SENTIMENT.includes(row.feature)) return;
    const brandKey = canonBrand(row.entity);
    if (!brandKey || !row.date) return;
    const key = featureIndexKey(row.date, brandKey);
    const current = featureNegByDateBrand.get(key) || { total: 0, value: 0 };
    current.total += row.total;
    current.value += row.value;
    featureNegByDateBrand.set(key, current);
    if (row.feature === 'top_stories_items' && row.total > 0 && row.value > 0) {
      topStoriesNegByBrandDay.set(key, (topStoriesNegByBrandDay.get(key) || 0) + row.value);
    }
  });

  featureControlRows.forEach((row) => {
    if (!FEATURE_ORDER_SENTIMENT.includes(row.feature)) return;
    const brandKey = canonBrand(row.entity);
    if (!brandKey || !row.date) return;
    const key = featureIndexKey(row.date, brandKey);
    const current = featureCtrlByDateBrand.get(key) || { total: 0, value: 0 };
    current.total += row.total;
    current.value += row.value;
    featureCtrlByDateBrand.set(key, current);
  });

  const crisisBrandKeys = new Set();
  topStoriesNegByBrandDay.forEach((negativeCount, key) => {
    if (negativeCount <= CRISIS_MIN_NEG) return;
    const parts = key.split('|');
    if (parts[1]) crisisBrandKeys.add(parts[1]);
  });

  return {
    featureNegByDateBrand,
    featureCtrlByDateBrand,
    crisisBrandKeys,
  };
}

function matchesCompanySize(flags = {}, companySizeFilter = 'all') {
  if (companySizeFilter === 'fortune500') return !!flags.f500;
  if (companySizeFilter === 'fortune1000') return !!flags.f1000;
  if (companySizeFilter === 'forbes') return !!flags.forbes;
  return true;
}

function riskScoreForRow(row) {
  const allSurfaceNeg = row.negSerpAll ?? row.negSerp ?? 0;
  const allSurfaceCtrl = row.ctrlPctAll ?? row.ctrlPct ?? 0;
  return (allSurfaceNeg * 0.55) + ((row.negNews ?? 0) * 0.35) + ((row.negSerp ?? 0) * 0.1) - (allSurfaceCtrl * 0.2);
}

function riskToneForRow(row) {
  const score = riskScoreForRow(row);
  if (score >= 0.58) return 'High';
  if (score >= 0.34) return 'Medium';
  return 'Low';
}

function labelForRange(count) {
  if (!count) return 'No sectors matched the current filters.';
  if (count === 1) return '1 sector matched the current filters.';
  return `${count} sectors matched the current filters.`;
}

class SectorStore {
  constructor(config) {
    this.config = config;
    this.coreByDays = new Map();
    this.rosterPromise = null;
    this.stockPromise = null;
    this.stockByBrand = new Map();
  }

  clear() {
    this.coreByDays.clear();
    this.rosterPromise = null;
    this.stockPromise = null;
    this.stockByBrand = new Map();
    parsedJsonCache.clear();
    latestDateCache.clear();
    clearSharedFetchCache();
  }

  async ensureRoster() {
    if (this.rosterPromise) return this.rosterPromise;
    this.rosterPromise = (async () => parseRoster(await fetchJson('/api/v1/roster')))();
    return this.rosterPromise;
  }

  async ensureCore(days) {
    const cacheKey = Number(days);
    if (this.coreByDays.has(cacheKey)) {
      return this.coreByDays.get(cacheKey);
    }

    const task = (async () => {
      const roster = await this.ensureRoster();
      const [countsRowsRaw, serpRowsRaw, featureRowsRaw, featureControlRowsRaw] = await Promise.all([
        fetchJson(`/api/v1/daily_counts?kind=brand_articles&days=${cacheKey}`),
        fetchJson(`/api/v1/daily_counts?kind=brand_serps&days=${cacheKey}`),
        fetchJson(`/api/v1/serp_features?entity=brand&days=${cacheKey}`),
        fetchJson(`/api/v1/serp_feature_controls?entity=brand&days=${cacheKey}`),
      ]);

      const countsRows = parseCountsRows(countsRowsRaw);
      const serpRows = parseSerpRows(serpRowsRaw);
      const featureRows = parseFeatureRows(featureRowsRaw, 'negative_count');
      const featureControlRows = parseFeatureRows(featureControlRowsRaw, 'controlled_count');
      const dates = collectDates(countsRows, serpRows, featureRows, featureControlRows);

      const countsByDate = buildNestedIndex(
        countsRows,
        'company',
        () => ({ positive: 0, neutral: 0, negative: 0 }),
        (current, row) => {
          current.positive += row.positive;
          current.neutral += row.neutral;
          current.negative += row.negative;
        },
      );

      const serpByDate = buildNestedIndex(
        serpRows,
        'company',
        () => ({ total: 0, negativeSerp: 0, controlled: 0 }),
        (current, row) => {
          current.total += row.total;
          current.negativeSerp += row.negativeSerp;
          current.controlled += row.controlled;
        },
      );

      const featureIndexes = buildFeatureIndexes(featureRows, featureControlRows);

      return {
        days: cacheKey,
        dates,
        countsRows,
        serpRows,
        featureRows,
        featureControlRows,
        countsByDate,
        serpByDate,
        ...roster,
        ...featureIndexes,
      };
    })();

    this.coreByDays.set(cacheKey, task);
    try {
      const core = await task;
      this.coreByDays.set(cacheKey, core);
      return core;
    } catch (error) {
      this.coreByDays.delete(cacheKey);
      throw error;
    }
  }

  async ensureStockData() {
    if (this.stockPromise) return this.stockPromise;
    this.stockPromise = (async () => {
      const latest = await fetchLatestDatedJson({
        key: `stock_data:${this.config.view}`,
        buildUrl: (dateStr) => `/api/v1/stock_data?date=${dateStr}`,
        maxDays: 7,
      });
      this.stockByBrand = parseStockRows(latest.rows);
      return this.stockByBrand;
    })();
    return this.stockPromise;
  }

  brandAllowed(core, brandKey, { companySizeFilter = 'all', crisisOnly = false, allowedSector = '' } = {}) {
    if (!brandKey) return false;
    if (allowedSector) {
      const brands = core.brandsBySector.get(allowedSector);
      if (!brands || !brands.has(brandKey)) return false;
    }
    if (!matchesCompanySize(core.fortuneFlags.get(brandKey), companySizeFilter)) return false;
    if (crisisOnly && !core.crisisBrandKeys.has(brandKey)) return false;
    return true;
  }

  buildRows(core, { date, companySizeFilter = 'all', crisisOnly = false } = {}) {
    const rows = [];
    const newsByBrand = core.countsByDate.get(date) || new Map();
    const serpByBrand = core.serpByDate.get(date) || new Map();

    core.brandsBySector.forEach((brandSet, sectorName) => {
      let positive = 0;
      let neutral = 0;
      let negative = 0;
      let serpTotal = 0;
      let serpNegative = 0;
      let serpControl = 0;
      let featureTotal = 0;
      let featureNegative = 0;
      let featureControl = 0;
      const dailyChanges = [];
      const sevenDayChanges = [];
      let brandCount = 0;
      let crisisBrandCount = 0;

      brandSet.forEach((brandKey) => {
        if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly })) return;
        brandCount += 1;
        if (core.crisisBrandKeys.has(brandKey)) crisisBrandCount += 1;

        const newsRow = newsByBrand.get(brandKey);
        if (newsRow) {
          positive += newsRow.positive;
          neutral += newsRow.neutral;
          negative += newsRow.negative;
        }

        const serpRow = serpByBrand.get(brandKey);
        if (serpRow) {
          serpTotal += serpRow.total;
          serpNegative += serpRow.negativeSerp;
          serpControl += serpRow.controlled;
        }

        const featureNeg = core.featureNegByDateBrand.get(featureIndexKey(date, brandKey));
        if (featureNeg) {
          featureTotal += featureNeg.total;
          featureNegative += featureNeg.value;
        }

        const featureCtrl = core.featureCtrlByDateBrand.get(featureIndexKey(date, brandKey));
        if (featureCtrl) {
          featureControl += featureCtrl.value;
        }

        const stock = this.stockByBrand.get(brandKey);
        if (stock?.dailyChange != null) dailyChanges.push(stock.dailyChange);
        if (stock?.sevenDayChange != null) sevenDayChanges.push(stock.sevenDayChange);
      });

      const newsTotal = positive + neutral + negative;
      const totalSurface = serpTotal + featureTotal;
      const row = {
        sector: sectorName,
        brandCount,
        crisisBrandCount,
        negNews: safeRatio(negative, newsTotal),
        negSerp: safeRatio(serpNegative, serpTotal),
        ctrlPct: safeRatio(serpControl, serpTotal),
        negSerpAll: safeRatio(serpNegative + featureNegative, totalSurface),
        ctrlPctAll: safeRatio(serpControl + featureControl, totalSurface),
        avgDailyChange: average(dailyChanges),
        avgStockChange: average(sevenDayChanges),
      };
      row.riskScore = riskScoreForRow(row);
      row.riskTone = riskToneForRow(row);

      if (newsTotal || serpTotal || featureTotal) {
        rows.push(row);
      }
    });

    return rows.sort((left, right) => left.sector.localeCompare(right.sector));
  }

  buildTimeline(core, { selectedSector = '', companySizeFilter = 'all', crisisOnly = false } = {}) {
    const dates = core.dates;
    const newsPos = [];
    const newsNeu = [];
    const newsNeg = [];
    const organicNeg = [];
    const organicCtrl = [];
    const allSurfaceNeg = [];
    const allSurfaceCtrl = [];

    dates.forEach((date) => {
      let positive = 0;
      let neutral = 0;
      let negative = 0;
      let serpTotal = 0;
      let serpNegative = 0;
      let serpControl = 0;
      let featureTotal = 0;
      let featureNegative = 0;
      let featureControl = 0;

      const newsByBrand = core.countsByDate.get(date) || new Map();
      newsByBrand.forEach((bucket, brandKey) => {
        if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly, allowedSector: selectedSector })) return;
        positive += bucket.positive;
        neutral += bucket.neutral;
        negative += bucket.negative;
      });

      const serpByBrand = core.serpByDate.get(date) || new Map();
      serpByBrand.forEach((bucket, brandKey) => {
        if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly, allowedSector: selectedSector })) return;
        serpTotal += bucket.total;
        serpNegative += bucket.negativeSerp;
        serpControl += bucket.controlled;
      });

      core.featureNegByDateBrand.forEach((bucket, key) => {
        const [rowDate, brandKey] = key.split('|');
        if (rowDate !== date) return;
        if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly, allowedSector: selectedSector })) return;
        featureTotal += bucket.total;
        featureNegative += bucket.value;
      });

      core.featureCtrlByDateBrand.forEach((bucket, key) => {
        const [rowDate, brandKey] = key.split('|');
        if (rowDate !== date) return;
        if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly, allowedSector: selectedSector })) return;
        featureControl += bucket.value;
      });

      const newsTotal = positive + neutral + negative;
      const surfaceTotal = serpTotal + featureTotal;
      newsPos.push((safeRatio(positive, newsTotal) || 0) * 100);
      newsNeu.push((safeRatio(neutral, newsTotal) || 0) * 100);
      newsNeg.push((safeRatio(negative, newsTotal) || 0) * 100);
      organicNeg.push((safeRatio(serpNegative, serpTotal) || 0) * 100);
      organicCtrl.push((safeRatio(serpControl, serpTotal) || 0) * 100);
      allSurfaceNeg.push((safeRatio(serpNegative + featureNegative, surfaceTotal) || 0) * 100);
      allSurfaceCtrl.push((safeRatio(serpControl + featureControl, surfaceTotal) || 0) * 100);
    });

    return {
      dates,
      newsPos,
      newsNeu,
      newsNeg,
      organicNeg,
      organicCtrl,
      allSurfaceNeg,
      allSurfaceCtrl,
    };
  }

  buildFeatureSeries(core, { selectedSector = '', companySizeFilter = 'all', crisisOnly = false, valueField = 'negative' } = {}) {
    const totalsByDate = new Map();
    const bucketByDate = new Map();
    const sourceRows = valueField === 'controlled' ? core.featureControlRows : core.featureRows;
    const serpValueField = valueField === 'controlled' ? 'controlled' : 'negativeSerp';

    core.serpRows.forEach((row) => {
      const brandKey = canonBrand(row.company);
      if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly, allowedSector: selectedSector })) return;
      if (!totalsByDate.has(row.date)) totalsByDate.set(row.date, 0);
      totalsByDate.set(row.date, totalsByDate.get(row.date) + row.total);
      if (!bucketByDate.has(row.date)) bucketByDate.set(row.date, {});
      const bucket = bucketByDate.get(row.date);
      if (!bucket.organic) bucket.organic = 0;
      bucket.organic += metricNumber(row[serpValueField]);
    });

    sourceRows.forEach((row) => {
      if (!FEATURE_ORDER_SENTIMENT.includes(row.feature)) return;
      const brandKey = canonBrand(row.entity);
      if (!this.brandAllowed(core, brandKey, { companySizeFilter, crisisOnly, allowedSector: selectedSector })) return;
      if (!totalsByDate.has(row.date)) totalsByDate.set(row.date, 0);
      totalsByDate.set(row.date, totalsByDate.get(row.date) + row.total);
      if (!bucketByDate.has(row.date)) bucketByDate.set(row.date, {});
      const bucket = bucketByDate.get(row.date);
      if (!bucket[row.feature]) bucket[row.feature] = 0;
      bucket[row.feature] += row.value;
    });

    const dates = core.dates.filter((date) => totalsByDate.has(date));
    const series = {};
    FEATURE_ORDER_SENTIMENT.forEach((feature) => {
      series[feature] = [];
    });

    dates.forEach((date) => {
      const total = totalsByDate.get(date) || 0;
      const bucket = bucketByDate.get(date) || {};
      FEATURE_ORDER_SENTIMENT.forEach((feature) => {
        const value = bucket[feature] || 0;
        series[feature].push(total ? (value / total) * 100 : 0);
      });
    });

    return { dates, series, order: FEATURE_ORDER_SENTIMENT };
  }

  buildSnapshot({ days, date, query = '', sectorFilter = '', companySizeFilter = 'all', crisisOnly = false, selectedSector = '' } = {}) {
    const core = this.coreByDays.get(days);
    if (!core || core instanceof Promise) {
      throw new Error('Sector core data is not ready yet.');
    }

    const activeDate = core.dates.includes(date) ? date : (core.dates[core.dates.length - 1] || '');
    const rows = activeDate ? this.buildRows(core, { date: activeDate, companySizeFilter, crisisOnly }) : [];
    const normalizedQuery = String(query || '').trim().toLowerCase();
    const filteredRows = rows.filter((row) => {
      if (sectorFilter && row.sector !== sectorFilter) return false;
      if (normalizedQuery && !row.sector.toLowerCase().includes(normalizedQuery)) return false;
      return true;
    });

    let resolvedSelectedSector = selectedSector;
    if (!resolvedSelectedSector || !filteredRows.some((row) => row.sector === resolvedSelectedSector)) {
      resolvedSelectedSector = filteredRows.length === 1 ? filteredRows[0].sector : '';
    }
    const selectedRow = filteredRows.find((row) => row.sector === resolvedSelectedSector) || null;
    const series = this.buildTimeline(core, {
      selectedSector: resolvedSelectedSector || sectorFilter,
      companySizeFilter,
      crisisOnly,
    });
    const featureSeries = this.buildFeatureSeries(core, {
      selectedSector: resolvedSelectedSector || sectorFilter,
      companySizeFilter,
      crisisOnly,
      valueField: 'negative',
    });
    const featureControlSeries = this.buildFeatureSeries(core, {
      selectedSector: resolvedSelectedSector || sectorFilter,
      companySizeFilter,
      crisisOnly,
      valueField: 'controlled',
    });

    const summary = {
      visibleSectorCount: filteredRows.length,
      visibleBrandCount: filteredRows.reduce((sum, row) => sum + row.brandCount, 0),
      avgNewsNeg: average(filteredRows.map((row) => row.negNews).filter((value) => value != null)),
      avgSurfaceNeg: average(filteredRows.map((row) => row.negSerpAll).filter((value) => value != null)),
      avgSurfaceCtrl: average(filteredRows.map((row) => row.ctrlPctAll).filter((value) => value != null)),
      hint: labelForRange(filteredRows.length),
    };

    const leaderboard = filteredRows
      .slice()
      .sort((left, right) => right.riskScore - left.riskScore)
      .slice(0, 8);

    return {
      core,
      activeDate,
      rows,
      filteredRows,
      selectedSector: resolvedSelectedSector,
      selectedRow,
      summary,
      series,
      featureSeries,
      featureControlSeries,
      leaderboard,
      sectorOptions: core.sectors,
      dateOptions: core.dates,
    };
  }
}

export function getSectorStore(config) {
  const key = `${config.view}:${config.tabId}`;
  if (!storeRegistry.has(key)) {
    storeRegistry.set(key, new SectorStore(config));
  }
  return storeRegistry.get(key);
}

export function clearAllSectorStores() {
  storeRegistry.forEach((store) => store.clear());
}
