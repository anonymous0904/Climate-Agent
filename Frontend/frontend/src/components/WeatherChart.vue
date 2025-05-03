<script lang="ts" setup>
import {ref, watch, computed, type Ref} from 'vue'
import axios from 'axios'
import {Bar, Line} from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title, Tooltip, Legend,
  LineElement, PointElement, CategoryScale, LinearScale, BarElement, BarController
} from 'chart.js'
import type {ChartOptions} from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'

ChartJS.register(Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale, BarElement,
  BarController, zoomPlugin)

interface WeatherData {
  Time: string
  ['Actual Value']: number
  ['Train Prediction']: number
}

const lineChart = ref<any>(null)

const props = defineProps<{
  variable: string,
  title: string
}>()

const categoricalMap: Record<string, string[]> = {
  present_fog: ['No fog', 'Fog'],
  precipitation: ['None', 'Rain', 'Snow'],
  cloud_nebulosity: ['Sky clear', 'Few', 'Scattered', 'Broken', 'Overcast']
}

const isCategorical = (v: string) => Object.prototype.hasOwnProperty.call(categoricalMap, v)


const chartData = ref<any>(null)

const chartOptions = computed<ChartOptions<'line'>>(() => ({
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
    },
    zoom: {
      pan: {
        enabled: true,
        mode: 'x',
        overScaleMode: 'x',
        threshold: 2
      },

      zoom: {
        wheel: {enabled: true},
        pinch: {enabled: true},
        mode: 'x',
        drag: {
          enabled: true,
          borderWidth: 1,
          backgroundColor: 'rgba(255,255,255,0.15)'
        },
        limits: {
          x: {min: 'original', max: 'original'},
          y: {min: 'original', max: 'original'}
        }
      }
    }
  },
  scales: {
    x: {
      ticks: {
        color: '#ccc',
        font: {size: 10},
        autoSkip: true,
        maxTicksLimit: 12,
        maxRotation: 30,
        minRotation: 0
      },
      title: {
        display: true,
        text: 'Time',
        color: '#ccc'
      }
    },
    y: isCategorical(props.variable)
      ? {
        ticks: {
          color: '#ccc',
          stepSize: 1,
          callback: (val: number | string) =>
            categoricalMap[props.variable][Number(val)] ?? val
        },
        suggestedMin: 0,
        suggestedMax: categoricalMap[props.variable].length - 1
      }
      : {
        ticks: {color: '#ccc'},
        title: {
          display: true,
          text: props.variable,
          color: '#ccc'
        }
      }
  }
}))

function buildDataset(label: string, values: number[], color: string, categorical: boolean) {
  return {
    label,
    data: values,
    borderColor: color,
    backgroundColor: categorical ? color + '55' : 'transparent',
    fill: false,
    tension: categorical ? 0 : 0.3,
    stepped: categorical ? true : false,
    type: categorical ? 'bar' : 'line'
  }
}


watch(() => props.variable, async (newVar) => {
  chartData.value = null
  try {
    const res = await axios.get<WeatherData[]>(`http://localhost:5000/api/${newVar}`)
    const data = res.data
    const categorical = isCategorical(newVar)

    chartData.value = {
      labels: data.map(d => d.Time),
      datasets: [
        buildDataset('Actual', data.map(d => d['Actual Value']), '#03DAC6', categorical),
        buildDataset('Prediction', data.map(d => d['Train Prediction']), '#BB86FC', categorical)
      ]
    }
  } catch (err) {
    console.error('Error fetching chart data:', err)
  }
}, {immediate: true})
</script>


<template>
  <div class="chart-container" v-if="chartData">

    <button
      class="reset-btn"
      @click="lineChart?.chart?.resetZoom()"
      title="Reset zoom">
      Reset zoom
    </button>

    <component
      class="chart-body"
      :is="isCategorical(variable) ? Bar : Line"
      ref="lineChart"
      :data="chartData"
      :options="chartOptions"
    />
  </div>
  <div v-else class="chart-loading">
    Loading data...
  </div>
</template>


<style scoped>

.chart-container {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #1e1e1e;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 1rem 2rem;
  border-radius: 12px;
  color: #fff;
  /*font-weight: 600;
  letter-spacing: .5px;*/
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

.reset-btn {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 5;

  padding: 2px 10px;
  font-size: .75rem;
  background: #333;
  border: 1px solid #555;
  color: #ccc;
  border-radius: 6px;
  cursor: pointer;
}

.reset-btn:hover {
  background: #444;
}

.chart-body {
  flex: 1 1 0;
  min-height: 0;
}

</style>
