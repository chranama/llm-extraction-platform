import { expect, test } from "@playwright/test";

test("app shell loads and top-level tabs switch", async ({ page }) => {
  await page.route(/\/(?:api\/)?v1\/models$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default_model: "demo-model",
        models: [],
        deployment_capabilities: { generate: true, extract: false },
      }),
    });
  });

  await page.route(/\/(?:api\/)?v1\/admin\/policy$/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ status: "allow", ok: true, enable_extract: false }),
    });
  });

  await page.route(/\/(?:api\/)?v1\/admin\/logs.*/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ total: 0, limit: 50, offset: 0, items: [] }),
    });
  });

  await page.goto("/");

  await expect(page.getByRole("heading", { name: "LLM Server" })).toBeVisible();
  await page.getByRole("button", { name: "Admin" }).click();
  await expect(page.getByRole("button", { name: "Load model" })).toBeVisible();

  await page.getByRole("button", { name: "Demo" }).click();
  await expect(page.getByRole("button", { name: /Generate clamp \(Demo A\)/i })).toBeVisible();
});
