<template>
  <div class="results-gallery">
    <div v-for="result in results" :key="result.productId" class="result-card">

      <!-- Card Header -->
      <div class="result-header">
        <div class="result-identity">
          <div class="result-id">
            <span class="id-hash">#</span>
            <span class="id-num">{{ result.productId.slice(-6).toUpperCase() }}</span>
          </div>
          <div class="result-tags">
            <span class="tag tag--cat">{{ getCategoryLabel(result.category) }}</span>
            <span class="tag tag--style">{{ getStyleLabel(result.style) }}</span>
            <span v-if="result.qualityEnabled" class="tag tag--quality">AI 评分</span>
          </div>
        </div>
        <div class="result-kpis">
          <div class="kpi">
            <div class="kpi-val">{{ result.retrievedCount }}</div>
            <div class="kpi-lbl">参考爆款</div>
          </div>
          <div class="kpi-divider"></div>
          <div class="kpi">
            <div class="kpi-val">{{ result.generatedCount }}</div>
            <div class="kpi-lbl">生成图片</div>
          </div>
        </div>
      </div>

      <!-- Generated Images -->
      <div class="generated-section">
        <div class="section-head">
          <h4 class="section-label">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <rect x="1" y="1" width="6" height="6" rx="1.5" stroke="#dc5a35" stroke-width="1.2"/>
              <rect x="9" y="1" width="6" height="6" rx="1.5" stroke="#dc5a35" stroke-width="1.2"/>
              <rect x="1" y="9" width="6" height="6" rx="1.5" stroke="#dc5a35" stroke-width="1.2"/>
              <rect x="9" y="9" width="6" height="6" rx="1.5" stroke="#dc5a35" stroke-width="1.2"/>
            </svg>
            生成的图片
          </h4>
          <span v-if="hasAllScores(result)" class="best-pill">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <path d="M6 1L7.5 4.5L11 5L8.5 7.5L9 11L6 9.5L3 11L3.5 7.5L1 5L4.5 4.5L6 1Z"/>
            </svg>
            最高分 {{ getHighestScore(result) }}/10
          </span>
        </div>

        <div class="generated-grid">
          <div
            v-for="(img, idx) in result.images"
            :key="'gen-' + idx"
            class="generated-item"
            :class="{ 'is-best': isBestScore(result, idx) }"
          >
            <!-- Best badge -->
            <div v-if="isBestScore(result, idx)" class="best-badge">
              <svg width="14" height="14" viewBox="0 0 12 12" fill="currentColor">
                <path d="M6 1L7.5 4.5L11 5L8.5 7.5L9 11L6 9.5L3 11L3.5 7.5L1 5L4.5 4.5L6 1Z"/>
              </svg>
              最佳
            </div>

            <!-- Image -->
            <div class="img-wrap">
              <img :src="img" :alt="`生成图 ${idx+1}`" class="gen-img">
            </div>

            <!-- Scores -->
            <div v-if="hasAllScores(result)" class="score-panel">
              <div class="score-total">
                <span class="score-num">{{ getImageScore(result, idx) }}</span>
                <span class="score-denom">/10</span>
                <span class="score-lbl">综合</span>
              </div>
              <div class="score-dims">
                <div class="dim" v-for="dim in scoreDims" :key="dim.key"
                  :class="{ 'dim--good': getScoreValue(result, idx, dim.key) >= 8 }">
                  <span class="dim-lbl">{{ dim.label }}</span>
                  <span class="dim-val">{{ getScoreValue(result, idx, dim.key) }}</span>
                </div>
              </div>
            </div>

            <!-- Actions -->
            <div class="item-actions">
              <a :href="img" :download="`generated_${idx + 1}.png`" class="action-btn">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M7 11V3M7 3L4 6M7 3L10 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M2 11H12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
                下载
              </a>
            </div>
          </div>
        </div>
      </div>

      <!-- Collapsible Reference Images -->
      <details class="collapsible">
        <summary class="collapsible-head">
          <span>参考爆款图</span>
          <svg class="chevron" width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </summary>
        <div class="ref-grid">
          <div class="ref-item">
            <div class="ref-label">原图</div>
            <div class="ref-wrap">
              <img :src="result.originalImage" alt="原图" class="ref-img">
            </div>
          </div>
          <div v-for="(refImg, idx) in result.referenceImages" :key="'ref-' + idx" class="ref-item">
            <div class="ref-label">参考 {{ idx + 1 }}</div>
            <div class="ref-wrap">
              <img :src="refImg" alt="参考图" class="ref-img">
            </div>
          </div>
        </div>
      </details>

      <!-- Collapsible Style Prompt -->
      <details class="collapsible">
        <summary class="collapsible-head">
          <span>风格分析 Prompt</span>
          <svg class="chevron" width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </summary>
        <div class="prompt-block">
          <p class="prompt-text">{{ result.stylePrompt }}</p>
        </div>
      </details>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  results: { type: Array, required: true }
})

const categoryLabels = {
  'midi_dress': '中长裙', 'maxi_dress': '长裙', 'mini_dress': '短裙',
  'skirt': '半身裙', 'top': '上装', 'pants': '裤装'
}
const styleLabels = {
  'casual': '休闲', 'formal': '正式', 'sporty': '运动',
  'elegant': '优雅', 'vintage': '复古', 'modern': '现代'
}
const scoreDims = [
  { key: 'clothing_accuracy', label: '服装' },
  { key: 'pose_naturalness', label: '姿势' },
  { key: 'scene_quality', label: '场景' },
  { key: 'lighting_quality', label: '布光' },
  { key: 'commercial_value', label: '商业' },
]

const getCategoryLabel = (c) => categoryLabels[c] || c
const getStyleLabel = (s) => styleLabels[s] || s

const getAllScores = (result) => {
  if (result.allScores?.length) return result.allScores
  if (result.qualityScores) return [result.qualityScores]
  return []
}
const hasAllScores = (result) => {
  const s = getAllScores(result)
  return s.length > 0 && s.length === result.images.length
}
const getImageScoreObj = (result, idx) => getAllScores(result)[idx] || {}
const getImageScore = (result, idx) => getImageScoreObj(result, idx).average || getImageScoreObj(result, idx).avg_score || '-'
const getScoreValue = (result, idx, dim) => getImageScoreObj(result, idx)[dim] || '-'
const getHighestScore = (result) => {
  const scores = getAllScores(result)
  const max = Math.max(...scores.map(s => s.average || s.avg_score || 0))
  return max.toFixed(1)
}
const isBestScore = (result, idx) => {
  const scores = getAllScores(result)
  if (!scores.length) return false
  const cur = getImageScore(result, idx)
  const max = Math.max(...scores.map(s => s.average || s.avg_score || 0))
  const curVal = typeof cur === 'number' ? cur : parseFloat(cur)
  return curVal === max && max > 0
}
</script>

<style scoped>
.results-gallery { display: grid; gap: 2rem; }

.result-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-light);
  border-radius: 24px;
  padding: 2rem;
  box-shadow: var(--shadow-sm);
  animation: fadeUp 0.4s ease-out;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Header */
.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.75rem;
  flex-wrap: wrap;
  gap: 1rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border-light);
}
.result-identity { display: flex; flex-direction: column; gap: 0.6rem; }
.result-id {
  display: flex;
  align-items: baseline;
  gap: 2px;
  font-family: 'Cormorant Garamond', Georgia, serif;
}
.id-hash { font-size: 1.2rem; color: var(--accent-coral); font-weight: 400; }
.id-num { font-size: 1.6rem; font-weight: 700; color: var(--text-primary); letter-spacing: 0.05em; }
.result-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.tag {
  padding: 3px 12px;
  border-radius: 100px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.03em;
}
.tag--cat { background: var(--bg-warm); color: var(--text-secondary); }
.tag--style { background: var(--accent-coral-pale); color: var(--accent-coral); }
.tag--quality { background: #f3f0ff; color: #7c3aed; }

.result-kpis { display: flex; align-items: center; gap: 1.25rem; }
.kpi { text-align: center; }
.kpi-val {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent-coral);
  line-height: 1;
}
.kpi-lbl { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }
.kpi-divider { width: 1px; height: 36px; background: var(--border-light); }

/* Generated Section */
.generated-section { margin-bottom: 1.5rem; }
.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}
.section-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}
.best-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  background: var(--accent-gold-light);
  border-radius: 100px;
  font-size: 0.78rem;
  font-weight: 600;
  color: #8a6420;
}

.generated-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 1rem;
}
.generated-item {
  background: var(--bg-base);
  border: 1.5px solid var(--border-light);
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.25s;
  position: relative;
}
.generated-item:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.generated-item.is-best {
  border-color: var(--accent-gold);
  box-shadow: 0 0 0 3px var(--accent-gold-light), var(--shadow-md);
}
.best-badge {
  position: absolute;
  top: 10px; right: 10px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  background: linear-gradient(135deg, var(--accent-gold), var(--accent-gold-light));
  border-radius: 100px;
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
  z-index: 5;
  animation: pulse-badge 2s ease-in-out infinite;
}
@keyframes pulse-badge {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.04); }
}

.img-wrap { width: 100%; }
.gen-img { width: 100%; aspect-ratio: 3/4; object-fit: cover; display: block; }

/* Scores */
.score-panel { padding: 0.875rem; background: var(--bg-surface); }
.score-total {
  display: flex;
  align-items: baseline;
  gap: 3px;
  margin-bottom: 0.6rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border-light);
}
.score-num { font-size: 1.5rem; font-weight: 700; color: var(--accent-gold); font-family: 'Cormorant Garamond', serif; }
.score-denom { font-size: 0.875rem; color: var(--text-muted); }
.score-lbl { font-size: 0.75rem; color: var(--text-muted); margin-left: 4px; }
.score-dims { display: flex; gap: 6px; }
.dim { display: flex; flex-direction: column; align-items: center; flex: 1; text-align: center; }
.dim-lbl { font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.dim-val { font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-top: 2px; }
.dim--good .dim-val { color: var(--accent-sage); }

/* Actions */
.item-actions {
  padding: 0.6rem 0.875rem;
  border-top: 1px solid var(--border-light);
  display: flex;
  justify-content: center;
}
.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  background: var(--bg-subtle);
  border-radius: 8px;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 0.8rem;
  font-weight: 500;
  transition: all 0.2s;
}
.action-btn:hover { background: var(--accent-coral-pale); color: var(--accent-coral); }

/* Collapsible */
.collapsible {
  background: var(--bg-base);
  border: 1px solid var(--border-light);
  border-radius: 14px;
  margin-bottom: 0.75rem;
  overflow: hidden;
}
.collapsible-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.25rem;
  cursor: pointer;
  user-select: none;
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-primary);
  list-style: none;
  transition: background 0.2s;
}
.collapsible-head:hover { background: var(--bg-subtle); }
.collapsible-head::-webkit-details-marker { display: none; }
.chevron { transition: transform 0.3s; color: var(--text-muted); }
.collapsible[open] .chevron { transform: rotate(180deg); }

.ref-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 0.75rem;
  padding: 0 1.25rem 1rem;
}
.ref-item { }
.ref-label {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}
.ref-wrap {
  aspect-ratio: 3/4;
  border-radius: 10px;
  overflow: hidden;
  background: var(--bg-subtle);
}
.ref-img { width: 100%; height: 100%; object-fit: cover; }

.prompt-block { padding: 0 1.25rem 1rem; }
.prompt-text { font-size: 0.9rem; line-height: 1.7; color: var(--text-secondary); }

@media (max-width: 640px) {
  .result-header { flex-direction: column; align-items: flex-start; }
  .result-kpis { width: 100%; justify-content: center; }
  .generated-grid { grid-template-columns: 1fr; }
}
</style>
