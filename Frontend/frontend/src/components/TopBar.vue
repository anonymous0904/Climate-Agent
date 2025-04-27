<template>
  <div class="top-bar">
    <Plane class="icon"/>
    <h2>{{ title }}</h2>
    <Star class="icon fav"/>
  </div>
</template>

<script lang="ts" setup>
import {Plane, Star} from 'lucide-vue-next'

defineProps<{ title: string }>()
</script>

<style scoped>

.icon {
  width: 26px;
  height: 26px;
}

.fav {
  margin-left: auto;
  opacity: .6;
  cursor: pointer;
}

.fav:hover {
  opacity: 1;
  color: gold;
}

.top-bar {
  display: flex;
  align-items: center;
  gap: .75rem;

  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.08);

  padding: 1rem 2rem;
  border-radius: 12px;
  color: #fff;
  font-weight: 600;
  letter-spacing: .5px;
}
</style>
