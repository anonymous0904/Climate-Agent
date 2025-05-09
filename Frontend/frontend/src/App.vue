<template>
  <div class="app-container">
    <TopBar class="top-bar" title="Cluj-Napoca International Airport"/>

    <div class="sidebar-shell">
      <div class="sidebar-scroll">
        <Sidebar @select="selected = $event"/>
      </div>
    </div>

    <div class="chart-shell">
      <WeatherChart
        :variable="selected"
        :title="variables[selected]"
      />
    </div>

  </div>
</template>

<script lang="ts" setup>
import {ref} from 'vue'
import Sidebar from './components/Sidebar.vue'
import WeatherChart from './components/WeatherChart.vue'
import TopBar from "@/components/TopBar.vue";

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
  display: grid;
  grid-template-columns: 200px 1fr;
  grid-template-rows: auto 1fr;

  height: 100%;
  width: 100%;
  position: relative;
  overflow: hidden;
  /* background-color: #121212; */
  color: white;
}

.app-container::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  background-image: radial-gradient(at 25% 20%, rgba(0, 180, 255, 0.25) 0%, transparent 60%),
  radial-gradient(at 80% 75%, rgba(178, 102, 255, 0.25) 0%, transparent 60%),
  linear-gradient(135deg, #0d0d17 0%, #1e1e2a 40%, #121212 100%);
  background-blend-mode: overlay;
  animation: gradientShift 20s ease-in-out infinite alternate,
  pulseDots 6s ease-in-out infinite alternate;
}


@keyframes pulseDots {
  0% {
    background-position: 25% 20%, 80% 75%;
  }
  100% {
    background-position: 30% 25%, 75% 70%;
  }
}

@keyframes gradientShift {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: 100% 100%;
  }
}

.top-bar {
  grid-column: 1 / -1;
  grid-row: 1;
}

.sidebar-shell {
  grid-column: 1;
  grid-row: 2;
  overflow: hidden;
  display: flex;
  border-top: 0;
  border-radius: 0 0 0 12px;
}

.sidebar-scroll {
  flex: 1;
}

.chart-shell {
  background: transparent;
  /*border: 1px solid rgba(255, 255, 255, .08);*/
  border-top: 0;
  border-radius: 12px;
  overflow: hidden;
}

.chart-scroll {
  flex: 1;
  overflow: auto;
  padding: 1rem 2rem 2rem; /* mic spațiu interior */
}

.content-container {
  grid-column: 2;
  grid-row: 2;
  display: flex;
  flex-direction: column;
  overflow: auto;
  padding: 0 1rem 1rem;
}

.sidebar-container {
  grid-column: 1;
  grid-row: 2;
  position: relative;
  overflow: hidden;
  /*background: #121212;*/
  /*overflow: auto; */
}

.chart-container {
  flex: 1;
  margin-top: 1rem;
  overflow: auto;
}
</style>

