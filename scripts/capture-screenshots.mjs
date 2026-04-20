/**
 * Capture 3 screenshots of the deployed site for the README.
 * Usage: node scripts/capture-screenshots.mjs
 */
import { chromium } from 'playwright';
import { join } from 'path';
import { mkdirSync, existsSync } from 'fs';

const SITE = 'https://search-benchmark-nu.vercel.app';
const OUT_DIR = join(import.meta.dirname, '..', 'docs', 'screenshots');
if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

async function waitReady(page) {
  // Wait for the loading screen to disappear (app becomes interactive)
  await page.waitForSelector('nav button:has-text("Search")', { timeout: 60000 });
  await page.waitForTimeout(500);
}

async function main() {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();

  console.log('Loading app (first run downloads model ~23MB)...');
  await page.goto(SITE, { waitUntil: 'networkidle', timeout: 120000 });
  await waitReady(page);

  // 1. About
  console.log('Screenshot: about.png');
  await page.click('nav button:has-text("About")');
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT_DIR, 'about.png'), fullPage: true });

  // 2. Search (pick a query and hit Search)
  console.log('Screenshot: search.png');
  await page.click('nav button:has-text("Search")');
  await page.waitForTimeout(500);

  const selectExists = await page.locator('select').count();
  if (selectExists > 0) {
    // Pick a medically-interesting query that will have clear results
    await page.selectOption('select', { index: 5 });
  } else {
    await page.fill('input[type="text"]', 'COVID-19 vaccines reduce severe disease');
  }
  await page.waitForTimeout(300);
  // Use the LAST button with "Search" text — nav button is first, submit button is second
  await page.locator('button', { hasText: /^Search$/ }).last().click();
  // Wait for semantic results to render (the slowest method)
  await page.waitForSelector('h3:has-text("Semantic")', { timeout: 120000 }).catch(() => {});
  await page.waitForSelector('text=ms', { timeout: 120000 }).catch(() => {});
  await page.waitForTimeout(5000); // buffer for all 3 columns to populate with results
  // Scroll to results section
  await page.evaluate(() => {
    const results = document.querySelector('h3');
    if (results) results.scrollIntoView({ behavior: 'instant', block: 'start' });
  });
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(OUT_DIR, 'search.png'), fullPage: true });

  // 3. Benchmark (run it)
  console.log('Screenshot: benchmark.png (running benchmark — takes a few minutes)...');
  await page.click('nav button:has-text("Benchmark")');
  await page.waitForTimeout(500);
  await page.click('button:has-text("Run Benchmark")');
  await page.waitForSelector('table tbody tr td:has-text("Hybrid")', { timeout: 600000 });
  await page.waitForTimeout(2000);
  // Force Recharts to render: scroll each chart into view, then dispatch resize events
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 400) {
      window.scrollTo(0, y);
      await new Promise((r) => setTimeout(r, 200));
    }
    window.scrollTo(0, 0);
    window.dispatchEvent(new Event('resize'));
    await new Promise((r) => setTimeout(r, 500));
    window.dispatchEvent(new Event('resize'));
  });
  await page.waitForTimeout(3000);
  // Verify charts have SVG rect elements (actual bars). If not, wait more.
  for (let attempt = 0; attempt < 10; attempt++) {
    const barCount = await page.evaluate(() =>
      document.querySelectorAll('.recharts-bar-rectangle, .recharts-bar path').length
    );
    if (barCount > 3) break;
    console.log(`  chart bars: ${barCount}, waiting...`);
    await page.evaluate(() => window.dispatchEvent(new Event('resize')));
    await page.waitForTimeout(1500);
  }
  await page.screenshot({ path: join(OUT_DIR, 'benchmark.png'), fullPage: true });

  await browser.close();
  console.log('Done:', OUT_DIR);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
