<template>
  <div class="upload-form">
    <div class="form-layout">
      <!-- Upload Area -->
      <div class="upload-col">
        <div
          class="upload-area"
          :class="{ 'dragover': isDragOver, 'has-file': previewImage }"
          @drop.prevent="handleDrop"
          @dragover.prevent="isDragOver = true"
          @dragleave.prevent="isDragOver = false"
          @click="selectFile"
        >
          <input ref="fileInput" type="file" accept="image/*" @change="handleFileSelect" style="display:none">

          <div v-if="!previewImage" class="upload-prompt">
            <div class="upload-icon-ring">
              <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                <path d="M20 28V12M20 12L13 19M20 12L27 19" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="20" cy="20" r="17" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3"/>
              </svg>
            </div>
            <h3 class="upload-title">拖拽图片到这里</h3>
            <p class="upload-hint">或点击选择文件，支持 JPG/PNG/WebP</p>
            <div class="upload-formats">
              <span class="format-tag">JPG</span>
              <span class="format-tag">PNG</span>
              <span class="format-tag">WebP</span>
            </div>
          </div>

          <div v-else class="upload-preview">
            <img :src="previewImage" alt="Preview" class="preview-image">
            <div class="preview-actions">
              <button class="preview-remove" @click.stop="removeFile">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                移除
              </button>
              <button class="preview-replace" @click.stop="selectFile">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M8 3H4a1 1 0 00-1 1v9a1 1 0 001 1h8a1 1 0 001-1V9M11 3l3 3-3 3M14 6H6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                更换
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Form Fields -->
      <div class="form-col">
        <div class="form-fields">
          <div class="form-group">
            <label class="form-label">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="1" width="12" height="12" rx="3" stroke="#dc5a35" stroke-width="1.2"/><path d="M4 7h6M4 5h3" stroke="#dc5a35" stroke-width="1.2" stroke-linecap="round"/></svg>
              商品品类
            </label>
            <select v-model="form.category" class="form-select" :disabled="loading">
              <option value="">选择品类</option>
              <option v-for="cat in categories" :key="cat.value" :value="cat.value">{{ cat.label }}</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="5.5" stroke="#dc5a35" stroke-width="1.2"/><path d="M5 7l2 2 3-3" stroke="#dc5a35" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
              商品风格
            </label>
            <select v-model="form.style" class="form-select" :disabled="loading">
              <option value="">选择风格</option>
              <option v-for="style in styles" :key="style.value" :value="style.value">{{ style.label }}</option>
            </select>
          </div>

          <div class="form-row-2">
            <div class="form-group">
              <label class="form-label">季节</label>
              <select v-model="form.season" class="form-select" :disabled="loading">
                <option value="all_season">全季</option>
                <option value="spring">春季</option>
                <option value="summer">夏季</option>
                <option value="autumn">秋季</option>
                <option value="winter">冬季</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">场景提示</label>
              <input v-model="form.sceneHint" type="text" class="form-input"
                placeholder="例如：温馨咖啡厅" :disabled="loading">
            </div>
          </div>

          <div class="advanced-toggle" @click="showAdvanced = !showAdvanced">
            <span>高级选项</span>
            <svg :class="{ rotated: showAdvanced }" width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
          </div>

          <div v-show="showAdvanced" class="advanced-panel">
            <label class="checkbox-label">
              <input type="checkbox" v-model="form.enableQualityCheck" :disabled="loading" class="checkbox">
              <span class="checkbox-custom"></span>
              <span class="checkbox-text">
                <span class="checkbox-title">启用 AI 质量评估</span>
                <span class="checkbox-desc">生成图片后进行多维度 AI 评分</span>
              </span>
            </label>
          </div>
        </div>

        <!-- Progress -->
        <div v-if="loading" class="progress-container">
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: progress + '%' }"></div>
          </div>
          <div class="progress-info">
            <span class="progress-text">{{ progressMessage }}</span>
            <span class="progress-pct">{{ progress }}%</span>
          </div>
          <div class="progress-steps">
            <div
              v-for="(step, i) in progressSteps"
              :key="i"
              class="progress-step"
              :class="{ active: progress > i * 20, done: progress > (i+1) * 20 }"
            >{{ step }}</div>
          </div>
        </div>

        <!-- Error -->
        <div v-if="error" class="error-alert">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="6.5" stroke="#dc5a35" stroke-width="1.2"/>
            <path d="M8 5v3M8 10.5v.5" stroke="#dc5a35" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          {{ error }}
        </div>

        <!-- Submit -->
        <button class="submit-btn" :disabled="!canSubmit || loading" @click="handleSubmit">
          <span v-if="!loading">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M9 2L9 16M9 16L4 11M9 16L14 11" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            生成宣传图
          </span>
          <span v-else class="loading-state">
            <span class="spinner-ring"></span>
            处理中...
          </span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['generated'])

const fileInput = ref(null)
const isDragOver = ref(false)
const previewImage = ref(null)
const selectedFile = ref(null)
const loading = ref(false)
const progress = ref(0)
const progressMessage = ref('')
const error = ref('')
const showAdvanced = ref(false)

const categories = ref([
  { value: 'midi_dress', label: '中长裙' },
  { value: 'maxi_dress', label: '长裙' },
  { value: 'mini_dress', label: '短裙' },
  { value: 'skirt', label: '半身裙' },
  { value: 'top', label: '上装' },
  { value: 'pants', label: '裤装' }
])
const styles = ref([
  { value: 'casual', label: '休闲' },
  { value: 'formal', label: '正式' },
  { value: 'sporty', label: '运动' },
  { value: 'elegant', label: '优雅' },
  { value: 'vintage', label: '复古' },
  { value: 'modern', label: '现代' }
])
const progressSteps = ['分析图片', '检索爆款', '风格分析', 'AI生图', '质量评分', '选择最优']

const form = ref({
  category: '',
  style: '',
  season: 'all_season',
  sceneHint: '',
  enableQualityCheck: false
})

const canSubmit = computed(() => selectedFile.value && form.value.category && form.value.style)

const selectFile = () => fileInput.value?.click()

const handleFileSelect = (e) => {
  const file = e.target.files?.[0]
  if (file && file.type.startsWith('image/')) setFile(file)
}

const handleDrop = (e) => {
  isDragOver.value = false
  const file = e.dataTransfer.files?.[0]
  if (file && file.type.startsWith('image/')) setFile(file)
}

const setFile = (file) => {
  selectedFile.value = file
  const reader = new FileReader()
  reader.onload = (e) => { previewImage.value = e.target.result }
  reader.readAsDataURL(file)
  error.value = ''
}

const removeFile = () => {
  selectedFile.value = null
  previewImage.value = null
  if (fileInput.value) fileInput.value.value = ''
}

const handleSubmit = async () => {
  if (!canSubmit.value) return
  loading.value = true
  error.value = ''
  progress.value = 0
  progressMessage.value = '上传图片中...'

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('category', form.value.category)
    formData.append('style', form.value.style)
    formData.append('season', form.value.season)
    formData.append('scene_hint', form.value.sceneHint)
    formData.append('enable_quality_check', form.value.enableQualityCheck ? 'true' : 'false')
    formData.append('retrieval_mode', 'two_stage')
    formData.append('sales_top_k', '100')
    formData.append('enable_multi_hop', 'true')
    formData.append('max_hops', '3')

    const response = await fetch('/api/upload', { method: 'POST', body: formData })
    if (!response.ok) throw new Error('上传失败')
    const data = await response.json()
    await pollTaskStatus(data.task_id, data.product_id)
  } catch (err) {
    error.value = err.message || '处理失败，请重试'
    loading.value = false
  }
}

const pollTaskStatus = async (taskId, productId) => {
  let attempts = 0
  let hasCompleted = false
  const maxAttempts = 120

  const poll = async () => {
    if (hasCompleted) return
    attempts++

    try {
      const response = await fetch(`/api/tasks/${taskId}`)
      const data = await response.json()
      progress.value = Math.round(data.progress * 100)

      if (data.status === 'pending') {
        progressMessage.value = '等待处理...'
      } else if (data.status === 'processing') {
        const messages = form.value.enableQualityCheck
          ? ['分析图片中...', '检索相似爆款...', '分析风格特征...', '生成多张图片...', 'AI 裁判评分中...', '选择最佳图片...']
          : ['分析图片中...', '检索相似爆款...', '分析风格特征...', '生成宣传图...']
        progressMessage.value = messages[Math.min(Math.floor(progress.value / 20), messages.length - 1)]
      } else if (data.status === 'completed' && !hasCompleted) {
        hasCompleted = true
        progressMessage.value = '生成完成！'
        loading.value = false
        removeFile()
        form.value = { category: '', style: '', season: 'all_season', sceneHint: '', enableQualityCheck: false }
        emit('generated', {
          productId: data.result.product_id,
          category: data.result.category,
          style: data.result.style,
          retrievedCount: data.result.retrieved_count,
          generatedCount: data.result.generated_count,
          stylePrompt: data.result.style_prompt,
          images: data.result.generated_images,
          originalImage: `/api/output/${data.result.product_id}/${data.result.product_id}_original.png`,
          referenceImages: [
            `/api/output/${data.result.product_id}/${data.result.product_id}_reference_1.png`,
            `/api/output/${data.result.product_id}/${data.result.product_id}_reference_2.png`,
            `/api/output/${data.result.product_id}/${data.result.product_id}_reference_3.png`
          ].filter((_, i) => i < data.result.retrieved_count),
          qualityScores: data.result.quality_scores || null,
          allScores: data.result.all_scores || [],
          qualityEnabled: data.result.quality_enabled || false
        })
        return
      } else if (data.status === 'failed') {
        hasCompleted = true
        throw new Error(data.error || '处理失败')
      }

      if (attempts < maxAttempts && !hasCompleted) {
        setTimeout(poll, 1000)
      } else if (!hasCompleted) {
        hasCompleted = true
        throw new Error('处理超时')
      }
    } catch (err) {
      hasCompleted = true
      error.value = err.message || '处理失败，请重试'
      loading.value = false
    }
  }
  poll()
}
</script>

<style scoped>
.upload-form {
  background: var(--bg-surface);
  border: 1px solid var(--border-light);
  border-radius: 24px;
  padding: 2.5rem;
  box-shadow: var(--shadow-md);
}

.form-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  align-items: start;
}

/* Upload Area */
.upload-area {
  border: 2px dashed var(--border);
  border-radius: 20px;
  padding: 3rem 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: var(--bg-base);
  min-height: 320px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.upload-area:hover {
  border-color: var(--accent-coral);
  background: var(--accent-coral-pale);
}
.upload-area.dragover {
  border-color: var(--accent-coral);
  background: var(--accent-coral-pale);
  transform: scale(1.01);
}
.upload-area.has-file {
  border-style: solid;
  padding: 1.5rem;
  border-color: var(--accent-sage-light);
  background: var(--bg-surface);
}

.upload-prompt { display: flex; flex-direction: column; align-items: center; gap: 1rem; }
.upload-icon-ring {
  width: 80px; height: 80px;
  border-radius: 50%;
  background: var(--accent-coral-pale);
  color: var(--accent-coral);
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.3s;
}
.upload-area:hover .upload-icon-ring { transform: scale(1.08); }
.upload-title { font-size: 1.2rem; font-weight: 600; color: var(--text-primary); }
.upload-hint { font-size: 0.875rem; color: var(--text-muted); }
.upload-formats { display: flex; gap: 8px; }
.format-tag {
  padding: 3px 10px;
  background: var(--bg-warm);
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 0.05em;
}

.upload-preview { width: 100%; }
.preview-image {
  max-width: 100%; max-height: 280px;
  border-radius: 14px; display: block; margin: 0 auto;
  box-shadow: var(--shadow-sm);
}
.preview-actions {
  display: flex; justify-content: center; gap: 8px; margin-top: 12px;
}
.preview-remove, .preview-replace {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 14px; border-radius: 8px; font-size: 0.8rem;
  font-weight: 500; cursor: pointer; border: none;
  transition: all 0.2s;
}
.preview-remove { background: #fee2e2; color: #dc2626; }
.preview-remove:hover { background: #fecaca; }
.preview-replace { background: var(--bg-subtle); color: var(--text-secondary); }
.preview-replace:hover { background: var(--bg-warm); }

/* Form */
.form-fields { display: flex; flex-direction: column; gap: 1.25rem; }
.form-group { display: flex; flex-direction: column; gap: 6px; }
.form-label {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.85rem; font-weight: 600;
  color: var(--text-secondary);
}
.form-row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

.form-select, .form-input {
  padding: 0.8rem 1rem;
  background: var(--bg-base);
  border: 1.5px solid var(--border);
  border-radius: 12px;
  color: var(--text-primary);
  font-size: 0.95rem;
  font-family: inherit;
  transition: all 0.25s;
  appearance: none;
}
.form-select {
  background-image: url("data:image/svg+xml,%3Csvg width='12' height='8' viewBox='0 0 12 8' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L6 7L11 1' stroke='%23a8a29e' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 36px;
}
.form-select:focus, .form-input:focus {
  outline: none;
  border-color: var(--accent-coral);
  background-color: var(--bg-surface);
  box-shadow: 0 0 0 3px rgba(220,90,53,0.08);
}
.form-select:disabled, .form-input:disabled { opacity: 0.5; cursor: not-allowed; }
.form-input::placeholder { color: var(--text-muted); }

.advanced-toggle {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.75rem 1rem;
  background: var(--bg-subtle);
  border-radius: 10px;
  cursor: pointer;
  font-size: 0.875rem; font-weight: 500;
  color: var(--text-secondary);
  transition: all 0.2s;
  user-select: none;
}
.advanced-toggle:hover { background: var(--bg-warm); }
.advanced-toggle svg { transition: transform 0.3s; color: var(--text-muted); }
.advanced-toggle svg.rotated { transform: rotate(180deg); }

.advanced-panel {
  background: var(--bg-subtle);
  border-radius: 12px;
  padding: 1rem;
  animation: fadeIn 0.2s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }

.checkbox-label {
  display: flex; align-items: flex-start; gap: 12px;
  cursor: pointer;
}
.checkbox { display: none; }
.checkbox-custom {
  width: 20px; height: 20px; min-width: 20px;
  border: 2px solid var(--border);
  border-radius: 6px;
  background: var(--bg-surface);
  display: flex; align-items: center; justify-content: center;
  transition: all 0.2s;
  margin-top: 1px;
}
.checkbox:checked + .checkbox-custom {
  background: var(--accent-coral);
  border-color: var(--accent-coral);
}
.checkbox:checked + .checkbox-custom::after {
  content: '';
  width: 6px; height: 10px;
  border: 2px solid white;
  border-top: none; border-left: none;
  transform: rotate(45deg) translateY(-1px);
}
.checkbox-title { display: block; font-size: 0.9rem; font-weight: 600; color: var(--text-primary); }
.checkbox-desc { display: block; font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }

/* Progress */
.progress-container { margin-top: 1.5rem; }
.progress-track {
  height: 6px; background: var(--bg-warm);
  border-radius: 3px; overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-coral), #f4a488);
  border-radius: 3px;
  transition: width 0.4s ease;
}
.progress-info {
  display: flex; justify-content: space-between;
  margin-top: 8px;
}
.progress-text { font-size: 0.8rem; color: var(--text-secondary); }
.progress-pct { font-size: 0.8rem; font-weight: 600; color: var(--accent-coral); }
.progress-steps {
  display: flex; gap: 4px; margin-top: 10px; flex-wrap: wrap;
}
.progress-step {
  font-size: 0.7rem; padding: 3px 8px;
  border-radius: 6px;
  background: var(--bg-warm);
  color: var(--text-muted);
  transition: all 0.3s;
}
.progress-step.active { background: var(--accent-coral-pale); color: var(--accent-coral); }
.progress-step.done { background: var(--accent-sage-light); color: var(--accent-sage); }

/* Error */
.error-alert {
  margin-top: 1rem;
  display: flex; align-items: center; gap: 8px;
  padding: 0.875rem 1rem;
  background: #fff4f0;
  border: 1px solid #fecaca;
  border-radius: 10px;
  color: #dc2626;
  font-size: 0.875rem;
}

/* Submit */
.submit-btn {
  width: 100%; margin-top: 1.5rem;
  padding: 1rem 2rem;
  background: var(--accent-coral);
  border: none; border-radius: 14px;
  color: white; font-size: 1rem; font-weight: 600;
  cursor: pointer; font-family: inherit;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  transition: all 0.25s;
  box-shadow: var(--shadow-warm);
}
.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(220,90,53,0.25);
}
.submit-btn:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none; }

.loading-state { display: flex; align-items: center; gap: 8px; }
.spinner-ring {
  width: 18px; height: 18px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 768px) {
  .form-layout { grid-template-columns: 1fr; gap: 2rem; }
  .upload-area { min-height: 240px; }
}
</style>
