<template>
  <div class="results-gallery">
    <div
      v-for="result in results"
      :key="result.productId"
      class="result-card"
    >
      <!-- Result Header -->
      <div class="result-header">
        <div class="result-info">
          <h3 class="result-title">生成结果 #{{ result.productId.slice(-6) }}</h3>
          <div class="result-meta">
            <span class="meta-tag">{{ result.category }}</span>
            <span class="meta-tag">{{ result.style }}</span>
          </div>
        </div>
        <div class="result-stats">
          <div class="stat">
            <span class="stat-value">{{ result.retrievedCount }}</span>
            <span class="stat-label">参考爆款</span>
          </div>
          <div class="stat">
            <span class="stat-value">{{ result.generatedCount }}</span>
            <span class="stat-label">生成图片</span>
          </div>
        </div>
      </div>

      <!-- Images Grid -->
      <div class="images-grid">
        <!-- Original Image -->
        <div class="image-item original">
          <div class="image-header">
            <span class="image-label">原图</span>
          </div>
          <div class="image-wrapper">
            <img :src="result.originalImage" alt="Original" class="result-image">
          </div>
        </div>

        <!-- Reference Images -->
        <div v-for="(refImg, idx) in result.referenceImages" :key="'ref-' + idx" class="image-item reference">
          <div class="image-header">
            <span class="image-label">参考 {{ idx + 1 }}</span>
          </div>
          <div class="image-wrapper">
            <img :src="refImg" alt="Reference" class="result-image">
          </div>
        </div>

        <!-- Generated Images -->
        <div v-for="(genImg, idx) in result.images" :key="'gen-' + idx" class="image-item generated">
          <div class="image-header">
            <span class="image-label generated-label">生成 {{ idx + 1 }}</span>
          </div>
          <div class="image-wrapper">
            <img :src="genImg" alt="Generated" class="result-image generated-image">
          </div>
          <div class="image-actions">
            <a :href="genImg" download class="action-btn">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 12V4M8 4L4 8M8 4L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                <path d="M4 14H12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
              </svg>
              下载
            </a>
          </div>
        </div>
      </div>

      <!-- Style Prompt -->
      <div class="style-prompt">
        <div class="prompt-header">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 1V3M8 13V15M3 8H1M15 8H13M3.514 3.514L4.929 4.929M11.071 11.071L12.485 12.485M3.514 12.486L4.929 11.071M11.071 4.929L12.485 3.514" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          <span>风格分析</span>
        </div>
        <p class="prompt-text">{{ result.stylePrompt }}</p>
      </div>

      <!-- Expand/Collapse -->
      <button
        class="expand-btn"
        @click="toggleExpand(result.productId)"
      >
        {{ expandedResults.has(result.productId) ? '收起' : '展开查看详情' }}
        <svg :class="{ rotated: expandedResults.has(result.productId) }" width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  results: {
    type: Array,
    required: true
  }
})

const expandedResults = ref(new Set())

const toggleExpand = (productId) => {
  if (expandedResults.value.has(productId)) {
    expandedResults.value.delete(productId)
  } else {
    expandedResults.value.add(productId)
  }
}
</script>

<style scoped>
.results-gallery {
  display: grid;
  gap: 2rem;
}

.result-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.5rem;
  animation: fadeUp 0.5s ease-out;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: 1.5rem;
  gap: 1rem;
  flex-wrap: wrap;
}

.result-info {
  flex: 1;
}

.result-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.result-meta {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.meta-tag {
  padding: 0.25rem 0.75rem;
  background: var(--primary);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 0.75rem;
  color: var(--text-muted);
}

.result-stats {
  display: flex;
  gap: 2rem;
}

.stat {
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* Images Grid */
.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.image-item {
  position: relative;
  background: var(--primary);
  border-radius: 12px;
  overflow: hidden;
}

.image-header {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  padding: 0.5rem 0.75rem;
  background: linear-gradient(to bottom, rgba(0,0,0,0.6), transparent);
  z-index: 1;
}

.image-label {
  font-size: 0.75rem;
  color: white;
  font-weight: 500;
}

.generated-label {
  color: var(--accent-light);
}

.image-wrapper {
  aspect-ratio: 3/4;
  overflow: hidden;
}

.result-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.image-item:hover .result-image {
  transform: scale(1.05);
}

.generated-image {
  border: 2px solid var(--accent);
}

.image-actions {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 0.75rem;
  background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
  display: flex;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.3s;
}

.image-item:hover .image-actions {
  opacity: 1;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  transition: transform 0.2s;
}

.action-btn:hover {
  transform: scale(1.05);
}

/* Style Prompt */
.style-prompt {
  background: var(--primary);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
}

.prompt-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  font-size: 0.875rem;
  color: var(--text-muted);
}

.prompt-text {
  font-size: 0.9375rem;
  line-height: 1.6;
  color: var(--text);
}

/* Expand Button */
.expand-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 10px;
  color: var(--text-muted);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.3s;
}

.expand-btn:hover {
  border-color: var(--accent);
  color: var(--text);
}

.expand-btn svg {
  transition: transform 0.3s;
}

.expand-btn svg.rotated {
  transform: rotate(180deg);
}

@keyframes fadeUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  .result-header {
    flex-direction: column;
  }

  .result-stats {
    width: 100%;
    justify-content: space-around;
  }

  .images-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
