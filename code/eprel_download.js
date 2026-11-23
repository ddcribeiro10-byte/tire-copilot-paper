// eprel_download.js — query used to build results/eprel_counts.json (run 2026-11-21).
// EPREL's public API is served from the EPREL website; run this in the browser console on
// https://eprel.ec.europa.eu/screen/product/tyres (same origin). It downloads all tyres in the three
// study sizes, keeps class C1, removes duplicate registrations of the same product
// (size, brand, model, load index, speed symbol), and prints the class cross-tabulation and noise statistics.
(async () => {
  const sizes = [[205, 55, 16], [225, 65, 17], [265, 70, 17]];
  const rows = [];
  for (const [w, a, d] of sizes) {
    for (let page = 1; ; page++) {
      const r = await fetch(`/api/products/tyres?_page=${page}&_limit=100&tyreSection=${w}&aspectRatio=${a}&rimDiameter=${d}`);
      const j = await r.json();
      for (const h of j.hits) {
        if (h.tyreClass !== 'C1') continue;
        rows.push({ size: w, rr: h.energyClass, wet: h.wetGripClass, noise: h.externalRollingNoiseValue,
                    snow: h.severeSnowTyre ? 1 : 0, brand: (h.supplierOrTrademark || '').trim(),
                    model: (h.modelIdentifier || '').trim(), li: h.loadCapacityIndex, ss: h.speedCategorySymbol });
      }
      if (page * 100 >= j.size) break;
    }
  }
  const seen = new Set(), U = [];
  for (const r of rows) { const k = [r.size, r.brand, r.model, r.li, r.ss].join('|'); if (!seen.has(k)) { seen.add(k); U.push(r); } }
  const tab = {}, noise = {};
  for (const r of U) {
    const key = r.size + '_' + (r.snow ? '3PMSF' : 'summer');
    (tab[key] = tab[key] || {})[r.rr + r.wet] = (tab[key][r.rr + r.wet] || 0) + 1;
    (noise[key] = noise[key] || []).push(+r.noise);
  }
  const ns = {};
  for (const k in noise) { const s = noise[k].sort((x, y) => x - y); ns[k] = [s.length, s[Math.floor(s.length * 0.1)], s[Math.floor(s.length * 0.5)], s[Math.floor(s.length * 0.9)]]; }
  console.log(JSON.stringify({ n_raw: rows.length, n_unique: U.length, tab, noise_n_p10_p50_p90: ns }));
})();
