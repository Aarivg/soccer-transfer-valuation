// export_squads.js — copy every Big 5 squad (club, value, contract…) from
// Transfermarkt to your clipboard, for the data refresh.
//
// Transfermarkt blocks GitHub's servers but loads fine in a normal browser,
// so the refresh starts here:
//   1. Open https://www.transfermarkt.com in Chrome
//   2. Open DevTools (Cmd+Option+J), paste this whole file into the Console,
//      press Enter, and wait ~4 minutes until it prints "Copied …"
//   3. On GitHub, add a file at data/raw/tm_browser_export.tsv on main, paste,
//      and commit. The "Data Refresh" workflow imports it, retrains the
//      model, commits the new data and removes the export file.
//
// It reads the same pages you'd open by hand (5 league pages + 96 squad
// pages), one every ~2–3 seconds.
(async () => {
  const SEASON = new Date().getMonth() >= 6 ? new Date().getFullYear()
                                            : new Date().getFullYear() - 1;
  const LEAGUES = { GB1: "premier-league", ES1: "laliga", L1: "bundesliga",
                    IT1: "serie-a", FR1: "ligue-1" };
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const clean = s => (s || "").replace(/[\t\n\r]+/g, " ").trim();
  const get = async path => {
    for (let attempt = 0; attempt < 3; attempt++) {
      const r = await fetch(path);
      const html = await r.text();
      if (r.status === 200 && html.includes('class="items"'))
        return new DOMParser().parseFromString(html, "text/html");
      console.warn(`retry ${attempt + 1} ${path}: HTTP ${r.status}`);
      await sleep(8000);
    }
    throw new Error(`could not load ${path}`);
  };

  const rows = [];
  let clubCount = 0;
  for (const [code, slug] of Object.entries(LEAGUES)) {
    const league = await get(`/${slug}/startseite/wettbewerb/${code}/saison_id/${SEASON}`);
    const clubs = {};
    league.querySelectorAll('table.items td.hauptlink a[href*="/startseite/verein/"]').forEach(a => {
      const id = a.getAttribute("href").match(/verein\/(\d+)/)[1];
      clubs[id] = clubs[id] || clean(a.getAttribute("title") || a.innerText);
    });
    for (const [clubId, clubName] of Object.entries(clubs)) {
      await sleep(1500 + Math.random() * 1500);
      const squad = await get(`/x/kader/verein/${clubId}/saison_id/${SEASON}/plus/1`);
      const table = squad.querySelector("table.items");
      const body = table.querySelector(":scope > tbody") || table;
      let n = 0;
      for (const tr of body.querySelectorAll(":scope > tr")) {
        const td = [...tr.querySelectorAll(":scope > td")];
        if (td.length < 10) continue;
        const a = td[1].querySelector('td.hauptlink a[href*="/profil/spieler/"]')
               || td[1].querySelector('a[href*="/profil/spieler/"]');
        if (!a) continue;
        const href = a.getAttribute("href").split("?")[0];
        const inner = td[1].querySelectorAll("tr");
        const img = td[1].querySelector("img");
        const src = img ? (img.getAttribute("data-src") || img.getAttribute("src") || "").split("?")[0] : "";
        const imgFile = (src.match(/portrait\/[a-z]+\/([^/]+)$/) || [])[1] || "";
        const crest = [...td[7].querySelectorAll("img")].map(i => clean(i.title))
                        .find(t => t && !t.startsWith(":")) || "";
        rows.push([
          href.match(/spieler\/(\d+)/)[1], clean(a.innerText), href.split("/")[1],
          inner.length > 1 ? clean(inner[1].innerText) : "",
          clean(td[2].innerText),
          [...td[3].querySelectorAll("img")].map(i => clean(i.title || i.alt)).join("|"),
          clean(td[4].innerText), clean(td[5].innerText), clean(td[6].innerText),
          crest, clean(td[8].innerText), clean(td[9].innerText), imgFile,
          clubId, clubName, code,
        ].join("\t"));
        n++;
      }
      clubCount++;
      console.log(`${code} ${clubName}: ${n} players (${clubCount} clubs done)`);
    }
  }
  copy(rows.join("\n"));  // DevTools console helper
  console.log(`Copied ${rows.length} players from ${clubCount} clubs to the clipboard.`);
})();
