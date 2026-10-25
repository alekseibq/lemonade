<script setup lang="ts">
import { computed, ref } from 'vue'

type State = { day: number; days_remaining?: number; capital: number; weather_forecast?: string; terminated: boolean }
type Result = { day: number; weather: string; price: number; produced: number; demand: number; sold: number; revenue: number; production_cost: number; profit: number; capital_after: number }
const api = '/api'
const sessionId = ref<string>()
const state = ref<State>()
const history = ref<Result[]>([])
const price = ref(20)
const production = ref(100)
const error = ref('')
const replayModel = ref('')
const replaySeed = ref(42)
const replayId = ref<string>()
const replaySteps = ref<{step: number; observation: number[]; action: {price: number; production: number}; result: Result; terminated: boolean}[]>([])
const canPlay = computed(() => !!sessionId.value && !state.value?.terminated)

async function newGame() {
  error.value = ''
  const response = await fetch(`${api}/sessions`, { method: 'POST' })
  if (!response.ok) return void (error.value = 'Не удалось начать игру')
  const data = await response.json()
  sessionId.value = data.session_id; state.value = data.state; history.value = []
}
async function playDay() {
  if (!sessionId.value) return
  const response = await fetch(`${api}/sessions/${sessionId.value}/actions`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({price: price.value, production: production.value}) })
  if (!response.ok) return void (error.value = 'Действие отклонено')
  const data = await response.json()
  history.value.push(data.result); state.value = data.state
}
async function startReplay() {
  const response = await fetch(`${api}/replays`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model_id: replayModel.value, seed: replaySeed.value}) })
  if (!response.ok) return void (error.value = 'Модель для replay недоступна')
  replayId.value = (await response.json()).replay_id; replaySteps.value = []
}
async function nextReplay() {
  if (!replayId.value) return
  const response = await fetch(`${api}/replays/${replayId.value}/next`, { method: 'POST' })
  if (!response.ok) return void (error.value = 'Replay завершён')
  replaySteps.value.push(await response.json())
}
</script>

<template>
  <main>
    <header><h1>🍋 Lemonade RL Lab</h1><button @click="newGame">Новая игра</button></header>
    <p v-if="!state">Начните эпизод: 10 дней, стартовый капитал 2 000 ₽.</p>
    <section v-else class="grid">
      <article><h2>День {{ state.day + 1 }} из 10</h2><p>Капитал: <b>{{ state.capital }} ₽</b></p><p v-if="!state.terminated">Прогноз: <b>{{ state.weather_forecast }}</b></p><p v-else>Эпизод завершён. Итог: <b>{{ state.capital - 2000 }} ₽</b></p></article>
      <form @submit.prevent="playDay"><h2>Решение</h2><label>Цена <input v-model.number="price" type="number" min="0" max="99"></label><label>Производство <input v-model.number="production" type="number" min="0" max="200"></label><button :disabled="!canPlay">Завершить день</button></form>
    </section>
    <p v-if="error" class="error">{{ error }}</p>
    <section class="replay"><h2>AI replay</h2><label>Model <input v-model="replayModel" placeholder="ppo-smoke"></label><label>Seed <input v-model.number="replaySeed" type="number" min="0"></label><button @click="startReplay" :disabled="!replayModel">Start</button><button @click="nextReplay" :disabled="!replayId">Next step</button><p v-if="replaySteps.length">Observation: {{ replaySteps.at(-1)?.observation }} · Action: {{ replaySteps.at(-1)?.action }} · Profit: {{ replaySteps.at(-1)?.result.profit }} ₽</p></section>
    <table v-if="history.length"><thead><tr><th>День</th><th>Погода</th><th>Цена</th><th>Продажи</th><th>Спрос</th><th>Прибыль</th></tr></thead><tbody><tr v-for="row in history" :key="row.day"><td>{{ row.day }}</td><td>{{ row.weather }}</td><td>{{ row.price }}</td><td>{{ row.sold }}</td><td>{{ row.demand }}</td><td>{{ row.profit }} ₽</td></tr></tbody></table>
  </main>
</template>
