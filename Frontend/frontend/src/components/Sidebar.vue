<template>
  <div class="sidebar">
    <button
      v-for="(title, key) in variables"
      :key="key"
      @click="$emit('select', key)"
    >
      {{ title }}
    </button>
  </div>
</template>

<script lang="ts" setup>
import {defineEmits} from 'vue'

const emit = defineEmits<{
  (e: 'select', value: string): void
}>()

const variables: Record<string, string> = {
  air_temperature: "Air Temperature",
  air_pressure: "Air Pressure",
  dew_point: "Dew Point",
  cloud_altitude: "Cloud Height",
  cloud_nebulosity: "Cloud Cover",
  present_fog: "Fog Presence",
  precipitation: "Precipitation",
  predominant_horizontal_visibility: "Visibility",
  wind_direction: "Wind Direction",
  wind_speed: "Wind Speed"
}
</script>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background-color: #121212;
}

button {
  background: transparent;
  color: #fff;
  border: 1px solid #444;
  margin: 0.25rem 0;
  padding: 0.5rem;
  cursor: pointer;
}

button:hover {
  background-color: #333;
}
</style>
