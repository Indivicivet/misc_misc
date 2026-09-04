import argparse
import html
import random
import sys
from pathlib import Path


def parse_input_file(filepath: Path) -> tuple[str, list[str]]:
    raw_text = filepath.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Input file '{filepath}' is empty.")

    # Split by the first blank line (one or more consecutive empty lines)
    blocks = [b.strip() for b in raw_text.split("\n\n", 1)]

    if len(blocks) == 1:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(lines) >= 9:
            return "", [clean_entry(line) for line in lines]
        return raw_text, []

    comments, entries_block = blocks[0], blocks[1]
    entries = []
    for line in entries_block.splitlines():
        cleaned = clean_entry(line)
        if cleaned:
            entries.append(cleaned)

    return comments, entries


def clean_entry(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith(("- ", "* ", "• ")):
        line = line[2:].strip()
    elif len(line) > 2 and line[0].isdigit() and line[1:3] in (". ", ") "):
        line = line[3:].strip()
    return line


def get_text_class(text: str) -> str:
    length = len(text)
    if length <= 20:
        return "len-short"
    if length <= 45:
        return "len-medium"
    if length <= 75:
        return "len-long"
    return "len-xlong"


def generate_cards(
    entries: list[str],
    card_count: int,
    grid_size: int,
    use_free_space: bool,
    seed: int | None = None,
) -> list[list[dict]]:
    total_cells = grid_size * grid_size
    has_free = use_free_space and (grid_size % 2 == 1)
    needed_per_card = total_cells - 1 if has_free else total_cells

    if len(entries) < needed_per_card:
        raise ValueError(
            f"Need at least {needed_per_card} entries for a {grid_size}x{grid_size} grid "
            f"{'with' if has_free else 'without'} free space, but only {len(entries)} entries were provided."
        )

    rng = random.Random(seed)
    cards = []
    center_idx = total_cells // 2

    for _ in range(card_count):
        sampled = rng.sample(entries, needed_per_card)
        card_cells = []
        sample_idx = 0

        for cell_idx in range(total_cells):
            if has_free and cell_idx == center_idx:
                card_cells.append({"is_free": True, "text": "FREE"})
            else:
                card_cells.append(
                    {
                        "is_free": False,
                        "text": sampled[sample_idx],
                        "class": get_text_class(sampled[sample_idx]),
                    }
                )
                sample_idx += 1
        cards.append(card_cells)

    return cards


def build_html(
    cards: list[list[dict]],
    comments: str,
    title: str,
    grid_size: int,
    cards_per_page: int,
) -> str:
    escaped_title = html.escape(title).strip()
    escaped_comments = "<br>".join(
        html.escape(line).strip() for line in comments.splitlines() if line.strip()
    )

    pages = []
    step = cards_per_page
    for i in range(0, len(cards), step):
        pages.append(cards[i : i + step])

    pages_html = []
    card_counter = 1

    for page_cards in pages:
        cards_markup = []
        for card_cells in page_cards:
            cells_html = []
            for cell in card_cells:
                if cell["is_free"]:
                    cells_html.append(
                        '<div class="grid-cell cell-free">'
                        '<div class="free-circle">FREE</div>'
                        "</div>"
                    )
                else:
                    text = html.escape(cell["text"])
                    cls = cell["class"]
                    cells_html.append(
                        f'<div class="grid-cell {cls}">'
                        f'<div class="cell-text">{text}</div>'
                        f"</div>"
                    )

            title_markup = (
                f'<div class="card-title">{escaped_title}</div>'
                if escaped_title
                else ""
            )
            comments_markup = (
                f'<div class="card-comments">{escaped_comments}</div>'
                if escaped_comments
                else ""
            )

            cards_markup.append(
                f"""<section class="bingo-card">
  <header class="card-header">
    <div class="header-left">
      {title_markup}
      {comments_markup}
    </div>
    <div class="card-badge">#{card_counter}</div>
  </header>
  <div class="grid-container" style="grid-template-columns: repeat({grid_size}, 1fr); grid-template-rows: repeat({grid_size}, 1fr);">
    {''.join(cells_html)}
  </div>
</section>"""
            )
            card_counter += 1

        card_content = "\n".join(cards_markup)
        pages_html.append(
            f'<div class="a4-page cards-per-page-{cards_per_page}">{card_content}</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escaped_title or "Bingo Cards"}</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 8mm;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #111;
      background-color: #f0f2f5;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .a4-page {{
      width: 194mm;
      height: 281mm;
      margin: 12mm auto;
      padding: 0;
      background: #fff;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      page-break-after: always;
      break-after: page;
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
      border-radius: 2px;
      overflow: hidden;
    }}
    .a4-page:last-child {{
      page-break-after: avoid;
      break-after: avoid;
    }}
    .bingo-card {{
      width: 100%;
      display: flex;
      flex-direction: column;
      padding: 2mm 3mm;
    }}
    .cards-per-page-1 .bingo-card {{
      height: 100%;
      padding: 4mm 5mm;
    }}
    .cards-per-page-2 .bingo-card {{
      height: 139mm;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 2mm;
      gap: 3mm;
    }}
    .header-left {{
      flex: 1;
    }}
    .card-title {{
      font-size: 14pt;
      font-weight: 800;
      line-height: 1.2;
      margin-bottom: 1.5mm;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .cards-per-page-1 .card-title {{
      font-size: 18pt;
    }}
    .card-comments {{
      font-size: 8.5pt;
      line-height: 1.25;
      color: #2b2b2b;
    }}
    .cards-per-page-1 .card-comments {{
      font-size: 11.5pt;
      line-height: 1.35;
      margin-bottom: 2mm;
    }}
    .card-badge {{
      font-size: 8pt;
      font-weight: 700;
      color: #666;
      background: #eee;
      border: 1px solid #ccc;
      padding: 1px 6px;
      border-radius: 3px;
      white-space: nowrap;
    }}
    .cards-per-page-1 .card-badge {{
      font-size: 9.5pt;
      padding: 2px 8px;
    }}
    .grid-container {{
      width: 100%;
      display: grid;
      border: 2px solid #111;
      background: #111;
      gap: 1.5px;
      flex: 1;
      aspect-ratio: 1 / 1;
      margin: 0 auto;
    }}
    .cards-per-page-2 .grid-container {{
      max-height: 122mm;
      max-width: 122mm;
    }}
    .cards-per-page-1 .grid-container {{
      max-height: 242mm;
      max-width: 190mm;
    }}
    .grid-cell {{
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 3px;
      overflow: hidden;
    }}
    .cell-text {{
      width: 100%;
      max-height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      hyphens: auto;
      word-break: break-word;
      overflow: hidden;
    }}
    .cards-per-page-2 .len-short {{ font-size: 10.5pt; font-weight: 600; line-height: 1.15; }}
    .cards-per-page-2 .len-medium {{ font-size: 9pt; font-weight: 500; line-height: 1.15; }}
    .cards-per-page-2 .len-long {{ font-size: 7.8pt; line-height: 1.1; }}
    .cards-per-page-2 .len-xlong {{ font-size: 6.8pt; line-height: 1.05; }}

    .cards-per-page-1 .len-short {{ font-size: 16.5pt; font-weight: 600; line-height: 1.2; }}
    .cards-per-page-1 .len-medium {{ font-size: 13.5pt; font-weight: 500; line-height: 1.2; }}
    .cards-per-page-1 .len-long {{ font-size: 11.5pt; line-height: 1.15; }}
    .cards-per-page-1 .len-xlong {{ font-size: 9.5pt; line-height: 1.1; }}

    .cell-free {{
      background-color: #fcfcfc;
    }}
    .free-circle {{
      width: 72%;
      aspect-ratio: 1 / 1;
      border: 2px solid #111;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 13pt;
      letter-spacing: 1px;
    }}
    .cards-per-page-1 .free-circle {{
      font-size: 22pt;
      border-width: 3px;
    }}

    @media print {{
      body {{
        background: none;
      }}
      .a4-page {{
        width: 100%;
        height: 100%;
        margin: 0;
        box-shadow: none;
        border-radius: 0;
      }}
    }}
  </style>
</head>
<body>
  {''.join(pages_html)}
  <script>
    function fitText() {{
      const cells = document.querySelectorAll('.grid-cell:not(.cell-free)');
      cells.forEach(cell => {{
        const textEl = cell.querySelector('.cell-text');
        if (!textEl) return;
        let fontSize = parseFloat(window.getComputedStyle(textEl).fontSize);
        while (
          (textEl.scrollHeight > cell.clientHeight - 4 || textEl.scrollWidth > cell.clientWidth - 4) &&
          fontSize > 5
        ) {{
          fontSize -= 0.5;
          textEl.style.fontSize = fontSize + 'px';
          textEl.style.lineHeight = '1.05';
        }}
      }});
    }}
    window.addEventListener('DOMContentLoaded', fitText);
    window.addEventListener('beforeprint', fitText);
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate print-ready A4 bingo cards from a text file."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("input.txt"),
        help="Path to the input text file (default: input.txt)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Grid size NxN (default: 5)",
    )
    parser.add_argument(
        "--card-count",
        type=int,
        default=10,
        help="Number of unique bingo cards to generate (default: 10)",
    )
    parser.add_argument(
        "--cards-per-page",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of cards per A4 page (default: 2)",
    )
    parser.add_argument(
        "--free-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include center free space (default: True)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Title header (default: empty, does not waste space)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("bingo_cards.html"),
        help="Output HTML filepath (default: bingo_cards.html)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible card generation",
    )

    args = parser.parse_args()

    input_path = args.input_file
    if not input_path.exists():
        fallback = Path("sample_input.txt")
        if input_path == Path("input.txt") and fallback.exists():
            input_path = fallback
        else:
            sys.exit(f"Error: Input file '{input_path}' not found.")

    try:
        comments, entries = parse_input_file(input_path)
    except Exception as e:
        sys.exit(f"Error parsing input file: {e}")

    if not entries:
        sys.exit(f"Error: No bingo entries found in '{input_path}'.")

    try:
        cards = generate_cards(
            entries=entries,
            card_count=args.card_count,
            grid_size=args.grid_size,
            use_free_space=args.free_space,
            seed=args.seed,
        )
    except ValueError as e:
        sys.exit(f"Error: {e}")

    html_content = build_html(
        cards=cards,
        comments=comments,
        title=args.title,
        grid_size=args.grid_size,
        cards_per_page=args.cards_per_page,
    )

    args.output.write_text(html_content, encoding="utf-8")
    print(
        f"Successfully generated {args.card_count} bingo cards "
        f"({(args.card_count + args.cards_per_page - 1) // args.cards_per_page} A4 pages) "
        f"to '{args.output}'."
    )


if __name__ == "__main__":
    main()
