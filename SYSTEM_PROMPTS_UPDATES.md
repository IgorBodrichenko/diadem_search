# System Prompts Updates - Chat Behavior Rules

## Summary of Changes

Все changes основаны на правилах из документа `правки.txt` - оптимизация поведения chatbot'а для более succinct, conversational и actionable ответов.

---

## 1. SYSTEM_PROMPT_CHAT (Q&A Chat Handler)

**Было:**
- Длинный verbose prompt с множеством деталей
- Упор на документы, Pinecone, process explanation
- Неявные структура и tone

**Стало:**
- Compact, direct, practical
- Убраны упоминания о документах и процессах
- Fokus на: natural language, INFORMATION-first, 1-2 вопроса макс
- 150-250 слов для Q&A ответов
- Special case для tactics questions (answer как live coach first)

**Ключевые правила:**
```
- Ask AT MOST ONE clarifying question
- Answer directly using INFORMATION
- 150-250 words max
- Use natural business language
```

---

## 2. SYSTEM_PROMPT_COACH_FINAL (Final Summary Generation)

**Было:**
- Verbose description of output formats
- Multiple optional sections and complex structure
- Неясный focus

**Стало:**
- Core rules простые и clear
- Output constraints: plain text, 200-400 words, 1 follow-up question max
- Special note для Q&A в final mode: answer directly, ask 1 question if needed
- Emphasis на coaching value, not repetition

**Ключевые правила:**
```
- Plain text only, 200-400 words max
- Ask AT MOST ONE follow-up question
- Challenge ambition, scrutinise positions
- Show empathy when pressure implied
```

---

## 3. MASTER_SYSTEM_PROMPT_TEXT (Template Filling Mode)

**Было:**
- Max 3 clarification questions
- 180 words default limit
- Complex rules about framework dumping

**Стало:**
- Max 1-2 questions per response (more aggressive cutoff)
- Succinct responses: 200-400 words for template fills
- Кореме: "Do NOT repeat or echo user input" - provide coaching instead
- New structure section: "Response structure for template filling"
  - YOUR WIN ZONE / THEIR WIN ZONE / TRADE-OFFS / ONE QUESTION
- Emphasis на naturalness, not framework pedagogy

**Ключевые правила:**
```
- Max 1-2 questions per response
- After 2 questions, MUST provide paste-ready output
- Don't echo user input - provide coaching value
- Structure for positions: YOUR / THEIR / TRADE-OFFS / QUESTION
```

---

## 4. LIMITS_POLICY (Interaction Limits)

**Было:**
```
- Keep responses <= 180 words
- Ask at most 2 clarifying questions
```

**Стало:**
```
- Q&A: 150-250 words max
- Template fills: 200-400 words max
- AT MOST 1-2 questions per response
- After 2 questions, generate output
```

---

## 5. SYSTEM_PROMPT_QA (Q&A Coaching - REWRITTEN)

**Было:**
- 60+ lines of verbose rules and definitions
- Repetitive position definitions
- Complex response style section

**Стало:**
- Concise "Core rules" section (main constraints)
- "For variable questions" (specific behavior)
- Position definitions only mentioned as "use ONLY when asked"
- Fokus na 150-250 words, 1 question max

**Structure:**
```
Core rules → For variable questions → Position definitions → INFORMATION empty/unrelated
```

---

## Key Patterns Applied

### 1. Question Limit: 1-2 Maximum
All prompts now enforce max 1-2 questions per response instead of 3. After 2 questions, assistant MUST generate output.

### 2. Word Limits by Context
- **Q&A**: 150-250 words
- **Template fills**: 200-400 words
- Not 180 words universal max

### 3. "Do NOT Repeat" Rule
Added explicitly in MASTER prompt: "Do NOT simply echo back what the user just said. Instead, provide coaching, challenge their thinking, suggest improvements."

### 4. Answer Directly First
Emphasize answering before asking questions. Don't start with questions.

### 5. Natural Business Language
Removed references to:
- Pinecone, documents, retrieval process
- Framework structure (unless user asks)
- Soft coaching language ("consider", "think about")

### 6. Template Filling Structure
For LOW/MID/HIGH positions, structure is now explicit:
```
YOUR WIN ZONE (with Low/Mid/High)
THEIR WIN ZONE (opposite direction)
TRADE-OFFS (what to ask for in return)
ONE QUESTION (move forward)
```

---

## Files Modified

- **app.py**:
  - `SYSTEM_PROMPT_CHAT` - Q&A handler (completely rewritten)
  - `SYSTEM_PROMPT_COACH_FINAL` - Final generation (reduced from ~40 lines to ~20)
  - `SYSTEM_PROMPT_QA` - Coaching (reduced from ~60 lines to ~30)
  - `LIMITS_POLICY` - Updated word/question limits
  - `MASTER_SYSTEM_PROMPT_TEXT` - Added response structure rules, reduced question limit

---

## Testing Checklist

- [ ] Chat Q&A responses are 150-250 words
- [ ] Template fills are 200-400 words
- [ ] Questions asked: max 1-2 per turn
- [ ] No echoing or repetition of user input
- [ ] Natural business tone (no "consider...", "think about...")
- [ ] Tactics questions answered as live coach first
- [ ] Position structures follow YOUR/THEIR/TRADE-OFFS/QUESTION format
- [ ] No mention of Pinecone, documents, or retrieval process
- [ ] After 2 questions, paste-ready output generated

---

## Impact Summary

These changes align the chatbot behavior with the "правки.txt" guidance document, making the system:
- ✅ More conversational and less verbose
- ✅ More actionable (fewer generic questions)
- ✅ More structured (clear position templates)
- ✅ More coachingoriented (challenge + suggest, not just ask)
- ✅ Quicker to output (max 2 questions instead of 3)
