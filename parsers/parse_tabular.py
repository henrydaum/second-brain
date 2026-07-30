"""Attachment parsing helpers for tabular inputs."""


dependencies_files = ['services/service_drive.py']
dependencies_pip = ['pandas']

# pandas and sqlite3 open the file themselves,
# so their actions cannot be turned into Requests. A process boundary is
# what actually contains them — and a malformed file that kills a box is a
# failed parse rather than a dead kernel.
isolation = "subprocess"

from guest.parsing import ParseResult, basename, register, suffix


# Returns standardized DataFrame object

"""
Tabular parsers.

Handles: CSV, TSV, XLSX, XLS, Parquet, Feather, SQLite.
Returns ParseResult(modality="tabular", tabular=pd.DataFrame).

Every tabular parser returns a real DataFrame — not a stringified
representation. Tasks that need text (like search indexing) can call
result.tabular.to_string() or generate their own text representation.
"""


DEFAULT_MAX_ROWS = 100_000  # safety limit for huge files


def _max_rows(config: dict) -> int:
    """Return max rows."""
    return config.get("max_rows", DEFAULT_MAX_ROWS)


# ===================================================================
# CSV / TSV
# ===================================================================

def parse_csv(sdk, path: str, config: dict = None) -> ParseResult:
    """Parse CSV/TSV into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        sdk.log("pandas not installed", level="debug")
        return ParseResult.failed("pandas not installed", modality="tabular")

    try:
        ext = suffix(path)
        sep = "\t" if ext == ".tsv" else ","
        limit = _max_rows(config)

        # Let pandas sniff the delimiter for CSV
        if ext != ".tsv":
            try:
                df = pd.read_csv(path, nrows=limit, sep=None, engine="python")
            except Exception:
                sdk.log(f"Failed to auto-detect CSV delimiter for {path}", level="debug")
                df = pd.read_csv(path, nrows=limit, sep=sep)
        else:
            df = pd.read_csv(path, nrows=limit, sep=sep)

        return ParseResult(
            modality="tabular",
            output={"default": df},
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="tabular")


register([".csv", ".tsv"], "tabular", parse_csv)


# ===================================================================
# XLSX / XLS
# ===================================================================

def parse_xlsx(sdk, path: str, config: dict = None) -> ParseResult:
    """Parse Excel files. Returns all sheets as a dict of DataFrames."""
    try:
        import pandas as pd
    except ImportError:
        sdk.log("pandas not installed", level="debug")
        return ParseResult.failed("pandas not installed", modality="tabular")

    try:
        limit = _max_rows(config)

        all_sheets = pd.read_excel(path, sheet_name=None, nrows=limit)

        sheet_meta = {}
        for name, df in all_sheets.items():
            sheet_meta[name] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }

        total_rows = sum(len(df) for df in all_sheets.values())

        return ParseResult(
            modality="tabular",
            output=all_sheets,
            metadata={
                "total_rows": total_rows,
                "sheet_count": len(all_sheets),
                "sheet_names": list(all_sheets.keys()),
                "sheets": sheet_meta,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="tabular")


register([".xlsx", ".xls"], "tabular", parse_xlsx)


# ===================================================================
# PARQUET / FEATHER
# ===================================================================

def parse_parquet(sdk, path: str, config: dict = None) -> ParseResult:
    """Parse Apache Parquet files into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        sdk.log("pandas not installed", level="debug")
        return ParseResult.failed("pandas not installed", modality="tabular")

    try:
        limit = _max_rows(config)
        ext = suffix(path)

        if ext == ".feather":
            df = pd.read_feather(path)
        else:
            df = pd.read_parquet(path)

        # Apply row limit after read (parquet doesn't support nrows natively)
        if len(df) > limit:
            df = df.head(limit)

        return ParseResult(
            modality="tabular",
            output={"default": df},
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "format": ext.lstrip("."),
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="tabular")


register([".parquet", ".feather"], "tabular", parse_parquet)


# ===================================================================
# SQLITE
# ===================================================================

def parse_sqlite(sdk, path: str, config: dict = None) -> ParseResult:
    """Parse sqlite."""
    try:
        import pandas as pd
        import sqlite3
    except ImportError as e:
        sdk.log(f"Missing dependency: {e}", level="debug")
        return ParseResult.failed(f"Missing dependency: {e}", modality="tabular")

    try:
        limit = _max_rows(config)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            conn,
        )["name"].tolist()

        if not tables:
            conn.close()
            return ParseResult.failed("No tables found in database", modality="tabular")

        all_tables = {}
        table_meta = {}
        total_rows = 0

        for table_name in tables:
            df = pd.read_sql(
                f'SELECT * FROM [{table_name}] LIMIT ?',
                conn,
                params=(limit,),
            )
            all_tables[table_name] = df
            total_rows += len(df)
            table_meta[table_name] = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }

        conn.close()

        return ParseResult(
            modality="tabular",
            output=all_tables,
            metadata={
                "total_rows": total_rows,
                "table_count": len(all_tables),
                "table_names": list(all_tables.keys()),
                "tables": table_meta,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse {path}: {e}", level="debug")
        return ParseResult.failed(str(e), modality="tabular")


register([".sqlite", ".db"], "tabular", parse_sqlite)


# ===================================================================
# GOOGLE SHEETS - if GoogleDriveService is unloaded, returns False
# ===================================================================


def parse_gsheet(sdk, path: str, config: dict = None) -> ParseResult:
    """
    Parse a .gsheet file (JSON shortcut) by downloading content from
    Google Drive as CSV and converting to a DataFrame.
    """
    import json
    import io

    try:
        import pandas as pd
    except ImportError:
        sdk.log("pandas not installed", level="debug")
        return ParseResult.failed("pandas not installed", modality="tabular")

    if not sdk.services.list().get("google_drive"):
        return ParseResult.failed(
            "Drive service not loaded — retry after loading",
            modality="tabular",
        )

    try:
        gsheet_data = json.loads(sdk.fs.read(path))

        doc_id = gsheet_data.get("doc_id")
        if not doc_id:
            return ParseResult.failed(
                "No doc_id found in .gsheet file", modality="tabular"
            )

        # Download as CSV via the Drive export API
        csv_text = sdk.services.call("google_drive", "download_csv", doc_id=doc_id)

        if csv_text is None:
            return ParseResult.failed(
                "Failed to download spreadsheet", modality="tabular"
            )

        if not csv_text.strip():
            return ParseResult.failed(
                "Spreadsheet is empty", modality="tabular"
            )

        # Parse CSV into DataFrame
        limit = _max_rows(config)
        df = pd.read_csv(io.StringIO(csv_text), nrows=limit)

        return ParseResult(
            modality="tabular",
            output={"default": df},
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "source": "google_drive",
                "doc_id": doc_id,
            },
        )
    except Exception as e:
        sdk.log(f"Failed to parse gsheet {basename(path)}: {e}", level="error")
        return ParseResult.failed(str(e), modality="tabular")


register(".gsheet", "tabular", parse_gsheet)
