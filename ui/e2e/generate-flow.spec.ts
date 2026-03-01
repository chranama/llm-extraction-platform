import { expect, test } from "@playwright/test";

test("generate flow works with mocked backend", async ({ page }) => {
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

  await page.route(/\/(?:api\/)?v1\/generate$/, async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    const prompt = String(body.prompt ?? "");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        model: "demo-model",
        cached: false,
        output: `echo:${prompt.slice(0, 20)}`,
      }),
    });
  });

  await page.goto("/");

  const promptInput = page.locator("textarea").first();
  await expect(promptInput).toBeVisible();
  await promptInput.fill("Playwright generate smoke test");
  await page.getByRole("button", { name: "Generate" }).last().click();

  await expect(page.getByText('"model": "demo-model"')).toBeVisible();
  await expect(page.getByText('"cached": false')).toBeVisible();
  await expect(page.getByText(/echo:Playwright generate/)).toBeVisible();
});
