<template>
  <div class="app-container">
    <div class="sidebar-container">
      <Sidebar @select="selected = $event"/>
    </div>
    <div class="chart-container">
      <WeatherChart :variable="selected" :title="variables[selected]"/>
    </div>
  </div>
</template>

<script lang="ts" setup>
import {ref} from 'vue'
import Sidebar from './components/Sidebar.vue'
import WeatherChart from './components/WeatherChart.vue'

const selected = ref<string>('air_temperature')

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

<style>
.app-container {
  display: flex;
  flex-direction: row;
  height: 100%;
  width: 100%;
  background-color: #121212;
  color: white;
}

.sidebar-container {
  width: 200px;
  background-color: #1e1e1e;
}

.chart-container {
  flex: 1;
  overflow: auto;
}
</style>

