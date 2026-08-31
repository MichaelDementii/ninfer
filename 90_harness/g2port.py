#!/usr/bin/env python3
"""Общий помощник переноса Г2 на upstream master.

Скрипты Г2 писались против camp/base, и якоря в них несут две вещи, которые на мастере другие:

1. **Контекст.** Прошлый круг правил те же места (например, считал токены в жадной ветке
   сэмплера), поэтому patch() принимает не одну пару (old, new), а список альтернатив: берётся
   первая, чей old встречается ровно один раз.

2. **Выравнивание.** Добавление переменных сдвигает колонку `=` у соседних строк, и якорь несёт
   уже новое выравнивание, которого в мастере нет. Поэтому при промахе точного сопоставления
   идёт вторая попытка — по регулярке, где прогоны пробелов внутри строки считаются
   эквивалентными. Требование единственности вхождения при этом сохраняется.

Дерево берётся из NINFER_TREE (по умолчанию /root/ninfer_d4).
"""
import os
import re
import sys

R = os.environ.get('NINFER_TREE', '/root/ninfer_d4')


def _loose_pattern(old):
    """Регулярка по old, где любой прогон пробелов/табов внутри строки — это [ \\t]+.

    Экранировать надо куски между прогонами, а не результат экранирования: re.escape в этом
    Python экранирует и сам пробел, и подстановка по уже экранированному тексту ломает шаблон.
    """
    return r'[ \t]+'.join(re.escape(part) for part in re.split(r'[ \t]+', old))


def _apply(s, old, new):
    """Возвращает (новый текст, как_совпало) или (None, причина)."""
    exact = s.count(old)
    if exact == 1:
        return s.replace(old, new, 1), 'точно'
    if exact > 1:
        return None, f'точных вхождений {exact}'
    matches = list(re.finditer(_loose_pattern(old), s))
    if len(matches) == 1:
        m = matches[0]
        return s[:m.start()] + new + s[m.end():], 'по пробелам'
    return None, f'точных 0, по пробелам {len(matches)}'


def patch(path, pairs):
    p = f'{R}/{path}'
    s = open(p, encoding='utf-8').read()
    loose = 0
    for entry in pairs:
        alts = entry if isinstance(entry, list) else [entry]
        reasons = []
        for old, new in alts:
            out, how = _apply(s, old, new)
            if out is not None:
                s = out
                loose += (how == 'по пробелам')
                break
            reasons.append(how)
        else:
            sys.exit(f'{path}: ни одна альтернатива не подошла ({"; ".join(reasons)}):\n'
                     f'{alts[0][0][:200]}')
    if not s.strip():
        sys.exit(f'отказ писать пустой {path}')
    open(p, 'w', encoding='utf-8').write(s)
    tail = f' (по пробелам: {loose})' if loose else ''
    print(f'пропатчен {path}{tail}')
