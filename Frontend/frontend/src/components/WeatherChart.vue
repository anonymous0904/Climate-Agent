<script lang="ts" setup>
import {ref, watch} from 'vue'
import {computed} from 'vue'
import axios from 'axios'
import {Line} from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title, Tooltip, Legend,
  LineElement, PointElement, CategoryScale, LinearScale
} from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale)

interface WeatherData {
  Time: string
  ['Actual Value']: number
  ['Train Prediction']: number
}

const props = defineProps<{
  variable: string,
  title: string
}>()

const chartData = ref<any>(null)

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: {
      display: true,
      text: props.title,
      color: '#fff',
      font: {size: 18}
    },
    legend: {
      labels: {
        color: '#fff'
      }
    }
  },
  scales: {
    x: {
      ticks: {color: '#ccc'},
      title: {
        display: true,
        text: 'Time',
        color: '#ccc'
      }
    },
    y: {
      ticks: {color: '#ccc'},
      title: {
        display: true,
        text: props.variable,
        color: '#ccc'
      }
    }
  }
}))

watch(() => props.variable, async (newVar) => {
  chartData.value = null
  try {
    const res = await axios.get<WeatherData[]>(`http://localhost:5000/api/${newVar}`)
    const data = res.data

    chartData.value = {
      labels: data.map(d => d.Time),
      datasets: [
        {
          label: 'Actual',
          data: data.map(d => d['Actual Value']),
          borderColor: '#03DAC6',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.3
        },
        {
          label: 'Prediction',
          data: data.map(d => d['Train Prediction']),
          borderColor: '#BB86FC',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.3
        }
      ]
    }
  } catch (err) {
    console.error('Error fetching chart data:', err)
  }
}, {immediate: true})
</script>


<template>
  <div class="chart-container" v-if="chartData">
    <Line :data="chartData" :options="chartOptions"/>
  </div>
  <div v-else class="chart-loading">
    Loading data...
  </div>
</template>


<style scoped>

.chart-container {
  flex-grow: 1;
  width: 100%;
  height: 100%;
  padding: 2rem;
  background-color: #1e1e1e;
}

.chart-loading {
  flex-grow: 1;
  padding: 2rem;
  color: #aaa;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #1e1e1e;
}

</style>
