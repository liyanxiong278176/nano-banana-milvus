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
          <h3 class="result-title">
            生成结果 #{{ result.productId.slice(-6) }}
            <span v-if="result.qualityEnabled" class="quality-badge">AI 评分模式</span>
          </h3>
          <div class="result-meta">
            <span class="meta-tag">{{ getCategoryLabel(result.category) }}</span>
            <span class="meta-tag">{{ getStyleLabel(result.style) }}</span>
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

      <!-- Generated Images with Scores -->
      <div class="generated-section">
        <div class="section-title-row">
          <h4 class="section-title">生成的图片</h4>
          <span v-if="hasAllScores(result)" class="best-hint">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 1L10 6L15 3M8 1L6 6L1 3M8 1V11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            最高分: {{ getHighestScore(result) }}/5
          </span>
        </div>
        <div class="generated-grid">
          <div
            v-for="(img, idx) in result.images"
            :key="'gen-' + idx"
            class="generated-item"
            :class="{ 'best-score': isBestScore(result, idx) }"
          >
            <!-- 星标 -->
            <div v-if="isBestScore(result, idx)" class="best-star">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/>
              </svg>
              <span>最佳</span>
            </div>

            <!-- 图片 -->
            <div class="generated-image-wrapper">
              <img :src="img" alt="Generated" class="generated-image">
            </div>

            <!-- 评分 -->
            <div v-if="hasAllScores(result)" class="image-scores">
              <div class="score-summary">
                <span class="total-score">{{ getImageScore(result, idx) }}/5</span>
                <span class="score-label">总分</span>
              </div>
              <div class="score-breakdown">
                <div class="mini-score" :class="{ good: getScoreValue(result, idx, 'clothing_accuracy') >= 4 }">
                  <span class="mini-label">服装</span>
                  <span class="mini-value">{{ getScoreValue(result, idx, 'clothing_accuracy') }}</span>
                </div>
                <div class="mini-score" :class="{ good: getScoreValue(result, idx, 'pose_naturalness') >= 4 }">
                  <span class="mini-label">姿势</span>
                  <span class="mini-value">{{ getScoreValue(result, idx, 'pose_naturalness') }}</span>
                </div>
                <div class="mini-score" :class="{ good: getScoreValue(result, idx, 'scene_quality') >= 4 }">
                  <span class="mini-label">场景</span>
                  <span class="mini-value">{{ getScoreValue(result, idx, 'scene_quality') }}</span>
                </div>
                <div class="mini-score" :class="{ good: getScoreValue(result, idx, 'lighting_quality') >= 4 }">
                  <span class="mini-label">布光</span>
                  <span class="mini-value">{{ getScoreValue(result, idx, 'lighting_quality') }}</span>
                </div>
                <div class="mini-score" :class="{ good: getScoreValue(result, idx, 'commercial_value') >= 4 }">
                  <span class="mini-label">商业</span>
                  <span class="mini-value">{{ getScoreValue(result, idx, 'commercial_value') }}</span>
                </div>
              </div>
            </div>

            <!-- 下载按钮 -->
            <div class="generated-actions">
              <a :href="img" :download="`generated_${idx + 1}.png`" class="download-btn">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M8 12V4M8 4L4 8M8 4L12 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                  <path d="M4 14H12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>

      <!-- Reference Images (Collapsible) -->
      <details class="collapsible-section">
        <summary class="collapsible-header">
          <span>参考爆款图</span>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </summary>
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
        </div>
      </details>

      <!-- Style Prompt -->
      <details class="collapsible-section">
        <summary class="collapsible-header">
          <span>风格分析</span>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </summary>
        <div class="style-prompt">
          <p class="prompt-text">{{ result.stylePrompt }}</p>
        </div>
      </details>

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
import { ref, computed } from 'vue'

const props = defineProps({
  results: {
    type: Array,
    required: true
  }
})

// 中文标签映射
const categoryLabels = {
  'midi_dress': '中长裙',
  'maxi_dress': '长裙',
  'mini_dress': '短裙',
  'skirt': '半身裙',
  'top': '上装',
  'pants': '裤装'
}

const styleLabels = {
  'casual': '休闲',
  'formal': '正式',
  'sporty': '运动',
  'elegant': '优雅',
  'vintage': '复古',
  'modern': '现代'
}

const getCategoryLabel = (category) => categoryLabels[category] || category
const getStyleLabel = (style) => styleLabels[style] || style

const expandedResults = ref(new Set())

const toggleExpand = (productId) => {
  if (expandedResults.value.has(productId)) {
    expandedResults.value.delete(productId)
  } else {
    expandedResults.value.add(productId)
  }
}

// 获取所有评分
const getAllScores = (result) => {
  if (result.allScores && Array.isArray(result.allScores) && result.allScores.length > 0) {
    return result.allScores
  }
  // 如果没有 allScores，但有 qualityScores，作为单张图片处理
  if (result.qualityScores) {
    return [result.qualityScores]
  }
  return []
}

// 是否有完整的评分数据
const hasAllScores = (result) => {
  const scores = getAllScores(result)
  return scores.length > 0 && scores.length === result.images.length
}

// 获取某张图片的评分对象
const getImageScoreObj = (result, idx) => {
  const scores = getAllScores(result)
  return scores[idx] || {}
}

// 获取某张图片的总分
const getImageScore = (result, idx) => {
  const scoreObj = getImageScoreObj(result, idx)
  return scoreObj.average || scoreObj.avg_score || '-'
}

// 获取某张图片某维度的分数
const getScoreValue = (result, idx, dimension) => {
  const scoreObj = getImageScoreObj(result, idx)
  return scoreObj[dimension] || '-'
}

// 获取最高分
const getHighestScore = (result) => {
  const scores = getAllScores(result)
  const maxScore = Math.max(...scores.map(s => s.average || s.avg_score || 0))
  return maxScore.toFixed(1)
}

// 判断是否是最高分
const isBestScore = (result, idx) => {
  const scores = getAllScores(result)
  if (scores.length === 0) return false

  const currentScore = getImageScore(result, idx)
  const maxScore = Math.max(...scores.map(s => s.average || s.avg_score || 0))
  const currentAvg = currentScore.average || currentScore.avg_score || 0

  return currentAvg === maxScore && maxScore > 0
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
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.quality-badge {
  padding: 0.25rem 0.75rem;
  background: linear-gradient(135deg, var(--accent), var(--accent-light));
  color: white;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 500;
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

/* Generated Section */
.generated-section {
  margin-bottom: 1.5rem;
}

.section-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.section-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--text);
}

.best-hint {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.1));
  border: 1px solid rgba(251, 191, 36, 0.3);
  border-radius: 20px;
  font-size: 0.875rem;
  color: #fbbf24;
}

.generated-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
}

.generated-item {
  background: var(--primary);
  border: 2px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s;
}

.generated-item.best-score {
  border-color: #fbbf24;
  box-shadow: 0 0 30px rgba(251, 191, 36, 0.3);
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.05), var(--primary));
}

.best-star {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.8rem;
  background: linear-gradient(135deg, #fbbf24, #f59e0b);
  border-radius: 20px;
  color: white;
  font-size: 0.875rem;
  font-weight: 600;
  z-index: 10;
  box-shadow: 0 2px 10px rgba(251, 191, 36, 0.4);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

.generated-image-wrapper {
  position: relative;
  width: 100%;
}

.generated-image {
  width: 100%;
  aspect-ratio: 3/4;
  object-fit: cover;
  display: block;
}

.image-scores {
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
}

.score-summary {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.total-score {
  font-size: 1.5rem;
  font-weight: 700;
  color: #fbbf24;
}

.score-label {
  font-size: 0.875rem;
  color: var(--text-muted);
}

.score-breakdown {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.5rem;
}

.mini-score {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.mini-label {
  font-size: 0.625rem;
  color: var(--text-muted);
  margin-bottom: 0.25rem;
}

.mini-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text);
}

.mini-score.good .mini-value {
  color: #10b981;
}

.generated-actions {
  padding: 0.75rem 1rem;
  display: flex;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
}

.download-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem 1rem;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s;
}

.download-btn:hover {
  background: var(--accent-light);
  transform: scale(1.05);
}

/* Collapsible Section */
.collapsible-section {
  background: var(--primary);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 1rem;
  overflow: hidden;
}

.collapsible-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  cursor: pointer;
  user-select: none;
  font-size: 0.9375rem;
  font-weight: 500;
  color: var(--text);
  transition: background 0.3s;
}

.collapsible-header:hover {
  background: var(--card-bg);
}

.collapsible-header svg {
  transition: transform 0.3s;
}

.collapsible-section[open] .collapsible-header svg {
  transform: rotate(180deg);
}

.collapsible-section > *:not(summary) {
  padding: 0 1.25rem 1rem;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
}

.image-item {
  position: relative;
  background: var(--card-bg);
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
}

/* Style Prompt */
.style-prompt {
  padding: 0 1.25rem 1rem;
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

  .generated-grid {
    grid-template-columns: 1fr;
  }

  .score-breakdown {
    grid-template-columns: repeat(5, 1fr);
  }
}
</style>
