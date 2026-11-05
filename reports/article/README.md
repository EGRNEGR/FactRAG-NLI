# Статья «Локальный контур RAG с независимой NLI-верификацией утверждений...» (итоговая версия)

| Файл | Что |
|---|---|
| `FactRAG-NLI_article.md` | исходник статьи (единый для .docx и LaTeX) |
| `references.md` | список источников по ГОСТ Р 7.0.5-2008; нумерация по первому упоминанию |
| `FactRAG-NLI_article.docx` | рукопись: A4, Times New Roman 14 пт, 1,5 интервала; формулы - объекты Word |
| `FactRAG-NLI_article.tex` | LaTeX-исходник (XeLaTeX: fontspec, polyglossia, unicode-math, STIX Two Math) |
| `figures/fig1-7.{png,pdf,svg}` | рисунки, 300 dpi, ширина 170 мм |
| `figure_data.json` | все данные рисунков, пересчитанные из логов |
| `FACT_CHECK.md` | источник каждой цифры и уточнения относительно задания |

Пересборка:

```bash
# 1. данные рисунков (окружение проекта, логи прогонов на ноутбуке; пути: ARTICLE_RUN_<ID>)
python scripts/article/figure_data.py
# 2. рисунки (любой Python с matplotlib)
python scripts/article/figures.py
# 3. .docx и .tex (python-docx, latex2mathml, lxml; MML2OMML.XSL из Microsoft Office)
python scripts/article/build_article.py --docx-copy "C:/Hakatons/FactRAG-NLI-3090/FactRAG-NLI_статья_итоговая.docx"
# 4. PDF из LaTeX
cd reports/article && xelatex FactRAG-NLI_article.tex
```
