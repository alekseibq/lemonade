import { expect, test } from '@playwright/test'

test('human can complete ten-day episode and inspect history', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Новая игра' }).click()
  await expect(page.getByText('День 1 из 10')).toBeVisible()

  const playDay = page.getByRole('button', { name: 'Завершить день' })
  for (let day = 1; day <= 10; day += 1) {
    await playDay.click()
    await expect(page.locator('tbody tr')).toHaveCount(day)
  }

  await expect(page.getByText('Эпизод завершён. Итог:')).toBeVisible()
  await expect(playDay).toBeDisabled()
})
