// Real browser regression: the actual React viewer inside its modal, served
// with production CSP headers. Backend calls are mocked, never sent to a kernel.
import { build } from 'vite';
import react from '@vitejs/plugin-react';
import { chromium, webkit, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

const caddy = await readFile('deploy/macos/Caddyfile', 'utf8');
const policies = [...caddy.matchAll(/header Content-Security-Policy "([^"]+)"/g)].map(m => m[1]);
const mainPolicy = policies.at(-1);
const appPolicy = policies.find(p => p.startsWith('sandbox allow-scripts;'));
const smoke = await readFile('docs/examples/html-app-diagnostic.html', 'utf8');
const result = await build({
  configFile: false, logLevel: 'error', publicDir: false,
  resolve: { alias: { '@': path.resolve('src') } },
  plugins: [react(), {
    name: 'html-viewer-harness',
    resolveId(id) { if (id === 'virtual:harness') return id; },
    load(id) {
      if (id !== 'virtual:harness') return;
      return `import React from 'react';
        import { createRoot } from 'react-dom/client';
        import { FileView } from '/src/components/file-view.tsx';
        import { appDocument } from '/src/lib/html-app.ts';
        import { Dialog, DialogContent, DialogTitle } from '/src/components/ui/dialog.tsx';
        createRoot(document.getElementById('root')).render(React.createElement(Dialog, {defaultOpen:true},
          React.createElement(DialogContent, {style:{width:'90vw',height:'90vh'}},
            React.createElement(DialogTitle, null, 'HTML test'),
            ${process.argv.includes('--expect-blocked')
              ? `React.createElement('iframe', {sandbox:'allow-scripts',srcDoc:appDocument(${JSON.stringify(smoke)},'legacy-token')})`
              : "React.createElement(FileView, {path:'/test.html',size:'full'})"})));`;
    },
  }],
  build: { write: false, rollupOptions: { input: 'virtual:harness', output: { codeSplitting: false } } },
});
const outputs = result.output;
const entry = outputs.find(o => o.type === 'chunk' && o.isEntry);
let calls = 0;
const server = createServer(async (req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Permissions-Policy', 'microphone=(self), camera=(), geolocation=()');
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/html-app-host.html') {
    res.setHeader('Content-Type', 'text/html');
    if (appPolicy) res.setHeader('Content-Security-Policy', appPolicy);
    res.end(await readFile('public/html-app-host.html', 'utf8'));
  } else if (url.pathname === '/files') {
    res.setHeader('Content-Security-Policy', 'sandbox');
    res.end(smoke);
  } else if (url.pathname.startsWith('/sdk/')) {
    calls++;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ data: { items: [], has_more: false, categories: [] } }));
  } else if (url.pathname === '/') {
    // Reproduce the attempted Mac fix as well as the original policy.
    res.setHeader('Content-Security-Policy', process.argv.includes('--attempted-fix')
      ? mainPolicy.replace("script-src 'self'", "script-src 'self' 'unsafe-inline'") : mainPolicy);
    res.setHeader('Content-Type', 'text/html');
    res.end(`<div id="root"></div><script type="module" src="/${entry.fileName}"></script>`);
  } else {
    const asset = outputs.find(o => '/' + o.fileName === url.pathname);
    res.setHeader('Content-Type', 'text/javascript');
    res.end(asset?.code ?? '');
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  const engine = process.argv.includes('--webkit') ? webkit : chromium;
  browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true });
  const messages = [];
  page.on('console', m => messages.push(m.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const frame = page.frameLocator('iframe');
  if (process.argv.includes('--expect-blocked')) {
    await expect(frame.locator('#boot')).toHaveText('BOOT PENDING');
    await frame.getByRole('button', { name: 'Test click' }).tap();
    await expect(frame.locator('#click')).toHaveText('CLICK PENDING');
    expect(messages.some(m => /Content Security Policy|content security policy/i.test(m))).toBe(true);
    console.log('REPRODUCED: author scripts and clicks blocked by inherited CSP.');
  } else {
    await expect(frame.locator('#boot')).toHaveText('SCRIPT EXECUTED — diagnostic v2');
    await frame.getByRole('button', { name: 'Test click' }).tap();
    await expect(frame.locator('#click')).toHaveText('CLICK RECEIVED');
    await frame.getByRole('button', { name: 'Test SDK' }).tap();
    await expect(frame.locator('#sdk')).toContainText('SDK OK');
    expect(calls).toBe(1);
    const isolated = await page.frames()[1].evaluate(() => {
      try { void parent.document.body; return false; } catch { return true; }
    });
    expect(isolated).toBe(true);
    // Oversized ordinary content should scroll naturally. The viewer must
    // also preserve scoped gesture controls and intentional App overflow CSS.
    const appFrame = page.frames()[1];
    await appFrame.evaluate(() => {
      const content = document.createElement('div');
      content.style.cssText = 'width:1600px;height:2000px';
      content.textContent = 'Oversized App content';
      document.body.append(content);
    });
    await page.locator('iframe').hover();
    await page.mouse.wheel(500, 600);
    await expect.poll(() => appFrame.evaluate(() => window.scrollY)).toBeGreaterThan(0);
    await expect.poll(() => appFrame.evaluate(() => window.scrollX)).toBeGreaterThan(0);
    const gestures = await appFrame.evaluate(() => {
      const control = document.createElement('div');
      control.style.touchAction = 'none';
      document.body.append(control);
      document.documentElement.style.overflow = 'hidden';
      return {
        control: getComputedStyle(control).touchAction,
        page: getComputedStyle(document.body).touchAction,
        overflow: getComputedStyle(document.documentElement).overflow,
      };
    });
    expect(gestures).toEqual({ control: 'none', page: 'auto', overflow: 'hidden' });
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(page.locator('iframe')).toHaveCount(0);
    console.log('PASS: script boot, touch, SDK relay, opaque origin, horizontal/vertical scrolling, modal close.');
  }
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
