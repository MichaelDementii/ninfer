#!/usr/bin/env python3
"""Пакет 09: перевод комментариев на английский.

Весь апстрим на английском; русские комментарии — отказ на входе. Смысл сохранён дословно,
строки уложены в 100 колонок.
"""
import io
import os
import re
import sys

R = os.environ.get('NINFER_TREE', '/root/ninfer_d4')

PAIRS = [
    # --- ядро приёмки -------------------------------------------------------------------------
    ("""// Добор из остатка (p - q)+ по опоре цели. q ищется в опоре черновика; токена, которого там
// нет, у черновика не было, значит q = 0. Если остаток вырожден (p <= q всюду по опоре),
// падаем на прежнее правило — это возможно только при численном вырождении.""",
     """// Resampling from the residual (p - q)+ over the target's support. q is looked up in the draft's
// support; a token that is not there was never proposed, so its q is zero. If the residual
// degenerates (p <= q everywhere on the support) the previous rule is used instead, which can only
// happen under numerical degeneracy."""),

    ("""                // q — вероятность того, что черновик предложит именно этот токен.
                // При жадном предложении q = 1, и правило вырождается в прежнее «u < p».
                // q записана для конкретного токена. Если строка конкурентности сменила
                // жильца, черновик придёт из хоста, а q останется от прежнего — тогда позиция
                // считается отклонённой, а замена берётся из полного p. Это обычный шаг без
                // спекуляции, то есть распределение сохраняется точно.""",
     """                // q is the probability that the draft proposes this very token. Under a
                // greedy proposal q is one and the rule degenerates into the previous "u < p".
                // q is recorded for one specific token. If the concurrency row changed tenants
                // the draft arrives from the host while q still belongs to the previous one; the
                // position is then treated as rejected and the replacement is drawn from the
                // full p. That is an ordinary non-speculative step, so the distribution is
                // preserved exactly."""),

    ("""                    // q — вероятность того, что черновик предложит именно этот токен.
                    // При жадном предложении q = 1, и правило вырождается в прежнее «u < p».
                    // q записана для конкретного токена. Если строка конкурентности сменила
                    // жильца, черновик придёт из хоста, а q останется от прежнего — тогда позиция
                    // считается отклонённой, а замена берётся из полного p. Это обычный шаг без
                    // спекуляции, то есть распределение сохраняется точно.""",
     """                    // q is the probability that the draft proposes this very token. Under
                    // a greedy proposal q is one and the rule degenerates into "u < p".
                    // q is recorded for one specific token. If the concurrency row changed
                    // tenants the draft arrives from the host while q still belongs to the
                    // previous one; the position is then treated as rejected and the replacement
                    // is drawn from the full p. That is an ordinary non-speculative step, so the
                    // distribution is preserved exactly."""),

    ("""                    // Остаток берём точным: (p - q)+ по опоре черновика. При одноточечном q
                    // это то же самое, что p с маской на предложенном токене.""",
     """                    // The residual is taken exactly: (p - q)+ over the draft's support. For
                    // a one-point q that is the same as p with the proposed token masked out."""),

    # --- раскладка буферов --------------------------------------------------------------------
    ("""    // Буфер q лежит по шагам: ne[0] = batch — непрерывный, ne[1] = число черновиков.""",
     """    // The q buffer is laid out by step: ne[0] = batch is contiguous, ne[1] is the draft count."""),

    ("""        // Первое измерение — максимум конкурентности, он может быть больше активных строк.""",
     """        // The first dimension is the concurrency maximum, which can exceed the active rows."""),

    # --- флаг --------------------------------------------------------------------------------
    ("""        // Включено по умолчанию: сэмплированное предложение черновика даёт приёмку
        // sum min(p, q) вместо E[p(argmax q)] и выигрывает на всех трёх целях
        // (+9.5% nvfp4-27B, +4.2% 35B-A3B, +1.4% 27B). Под --greedy правило
        // вырождается в прежнее побитово. Выключатель: NINFER_MTP_SAMPLED_DRAFT=0.""",
     """        // On by default: sampling the draft proposal gives an acceptance of sum min(p, q)
        // instead of E[p(argmax q)] and wins on all three targets (+9.5% nvfp4-27B,
        // +4.2% 35B-A3B, +1.4% 27B). Under --greedy the rule degenerates into the previous one
        // bit for bit. The switch is NINFER_MTP_SAMPLED_DRAFT=0."""),

    # --- предложение --------------------------------------------------------------------------
    ("""// Черновик берётся из собственного усечённого распределения головы предложений, а не из её
// argmax. Одноточечное предложение вырождает правило Метрополиса в «принять с вероятностью
// p(x*)»; полноценное q даёт ожидаемую приёмку sum_x min(p, q), что для откалиброванной головы
// заметно больше. Возвращаем и q(выбранного) — ядру приёмки оно нужно для min(1, p/q).""",
     """// The draft is drawn from the proposal head's own truncated distribution rather than from its
// argmax. A one-point proposal collapses the Metropolis rule into "accept with probability p(x*)";
// a full q gives an expected acceptance of sum_x min(p, q), which is markedly larger for a
// calibrated head. q of the chosen token is returned as well: the acceptance kernel needs it for
// min(1, p/q)."""),

    ("""        // Опора приходит в номерах строк головы предложений — переводим её в идентификаторы
        // токенов, иначе сравнивать её с опорой цели нельзя.""",
     """        // The support arrives as proposal-head row numbers; it is translated into token ids,
        // because otherwise it cannot be compared with the target's support."""),

    ("""        // q(черновика) по шагам: раскладка [batch][drafts], как у next_drafts.""",
     """        // The draft's q by step: laid out [batch][drafts], the same as next_drafts."""),

    ("""        // Опора черновика по шагам: 20 кандидатов на строку, раскладка [cap*batch][drafts].""",
     """        // The draft's support by step: 20 candidates per row, laid out [cap*batch][drafts]."""),

    ("""        // Токен, для которого записана q: приёмка сверяет его с проверяемым черновиком.""",
     """        // The token q was recorded for: acceptance checks it against the verified draft."""),
]

CYRILLIC = re.compile(r'[Ѐ-ӿ]')

files = []
for root, _dirs, names in os.walk(os.path.join(R, 'src')):
    for n in names:
        if n.endswith(('.cu', '.cuh', '.cpp', '.h')):
            files.append(os.path.join(root, n))
for root, _dirs, names in os.walk(os.path.join(R, 'include')):
    for n in names:
        if n.endswith(('.h', '.cuh')):
            files.append(os.path.join(root, n))

applied = {i: 0 for i in range(len(PAIRS))}
for path in files:
    s = io.open(path, encoding='utf-8').read()
    if not CYRILLIC.search(s):
        continue
    original = s
    for i, (old, new) in enumerate(PAIRS):
        n = s.count(old)
        if n:
            s = s.replace(old, new)
            applied[i] += n
    if s != original:
        io.open(path, 'w', encoding='utf-8', newline='\n').write(s)
        print('переведён ' + os.path.relpath(path, R))

missing = [i for i, n in applied.items() if n == 0]
if missing:
    print('НЕ НАЙДЕНЫ блоки: ' + ', '.join(str(i) for i in missing))

left = []
for path in files:
    s = io.open(path, encoding='utf-8').read()
    for k, line in enumerate(s.split('\n'), 1):
        if CYRILLIC.search(line):
            left.append('%s:%d: %s' % (os.path.relpath(path, R), k, line.strip()[:90]))
print('осталось кириллических строк: %d' % len(left))
for l in left[:20]:
    print('  ' + l)
