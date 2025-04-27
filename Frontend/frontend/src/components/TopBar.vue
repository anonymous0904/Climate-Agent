<template>
  <div class="top-bar">
    <h2>{{ title }}</h2>
  </div>
</template>

<script lang="ts" setup>
defineProps<{ title: string }>()
</script>

<style scoped>
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
