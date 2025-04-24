<script lang="ts" setup>
import {ref, watch} from 'vue'
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

watch(() => props.variable, async (newVar) => {
  const res = await axios.get<WeatherData[]>(`http://localhost:5000/api/${newVar}`)
  const data = res.data

  chartData.value = {
    labels: data.map(d => d.Time),
    datasets: [
      {
        label: 'Actual',
        data: data.map(d => d['Actual Value']),
        borderColor: '#03DAC6',
        fill: false,
        tension: 0.3
      },
      {
        label: 'Prediction',
        data: data.map(d => d['Train Prediction']),
        borderColor: '#BB86FC',
        fill: false,
        tension: 0.3
      }
    ]
  }
}, {immediate: true})
</script>


<template>

</template>

<style scoped>

</style>
