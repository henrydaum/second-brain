import logging
from pathlib import Path
from Stage_1.ParseResult import ParseResult
import Stage_1.registry as registry

logger = logging.getLogger(__name__)

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
    return config.get("max_rows", DEFAULT_MAX_ROWS)


# ===================================================================
# CSV / TSV
# ===================================================================

def parse_csv(path: str, config: dict) -> ParseResult:
    """Parse CSV/TSV into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        logger.debug("pandas not installed")
        return ParseResult.failed("pandas not installed", modality="tabular")

    try:
        ext = Path(path).suffix.lower()
        sep = "\t" if ext == ".tsv" else ","
        limit = _max_rows(config)

        # Let pandas sniff the delimiter for CSV
        if ext != ".tsv":
            try:
                df = pd.read_csv(path, nrows=limit, sep=None, engine="python")
            except Exception:
                logger.debug(f"Failed to auto-detect CSV delimiter for {path}")
                df = pd.read_csv(path, nrows=limit, sep=sep)
        else:
            df = pd.read_csv(path, nrows=limit, sep=sep)

        return ParseResult(
            modality="tabular",
            output=df,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="tabular")


registry.register([".csv", ".tsv"], "tabular", parse_csv)


# ===================================================================
# XLSX / XLS
# ===================================================================

def parse_xlsx(path: str, config: dict) -> ParseResult:
    """Parse Excel files into a DataFrame (first sheet by default)."""
    try:
        import pandas as pd
    except ImportError:
        logger.debug("pandas not installed")
        return ParseResult.failed("pandas not installed", modality="tabular")

    try:
        limit = _max_rows(config)
        sheet_name = config.get("sheet_name", 0)  # default: first sheet

        df = pd.read_excel(path, sheet_name=sheet_name, nrows=limit)

        # Get all sheet names for metadata
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True)
            sheet_names = wb.sheetnames
            wb.close()
        except Exception:
            logger.debug("openpyxl not installed")
            sheet_names = []

        return ParseResult(
            modality="tabular",
            output=df,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sheet_names": sheet_names,
                "active_sheet": sheet_name,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="tabular")


registry.register([".xlsx", ".xls"], "tabular", parse_xlsx)


# ===================================================================
# PARQUET / FEATHER
# ===================================================================

def parse_parquet(path: str, config: dict) -> ParseResult:
    """Parse Apache Parquet files into a DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        logger.debug("pandas not installed")
        return ParseResult.failed("pandas not installed", modality="tabular")

    try:
        limit = _max_rows(config)
        ext = Path(path).suffix.lower()

        if ext == ".feather":
            df = pd.read_feather(path)
        else:
            df = pd.read_parquet(path)

        # Apply row limit after read (parquet doesn't support nrows natively)
        if len(df) > limit:
            df = df.head(limit)

        return ParseResult(
            modality="tabular",
            output=df,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "format": ext.lstrip("."),
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="tabular")


registry.register([".parquet", ".feather"], "tabular", parse_parquet)


# ===================================================================
# SQLITE
# ===================================================================

def parse_sqlite(path: str, config: dict) -> ParseResult:
    """
    Parse a SQLite database into a DataFrame.

    By default reads the first table. Specify config["table_name"]
    to target a specific table.
    """
    try:
        import pandas as pd
        import sqlite3
    except ImportError as e:
        logger.debug(f"Missing dependency: {e}")
        return ParseResult.failed(f"Missing dependency: {e}", modality="tabular")

    try:
        limit = _max_rows(config)

        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

        # Get all table names
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            conn,
        )["name"].tolist()

        if not tables:
            conn.close()
            return ParseResult.failed("No tables found in database", modality="tabular")

        # Read the target table
        table_name = config.get("table_name", tables[0])
        df = pd.read_sql(
            f'SELECT * FROM [{table_name}] LIMIT ?',
            conn,
            params=(limit,),
        )

        conn.close()

        return ParseResult(
            modality="tabular",
            output=df,
            metadata={
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "tables": tables,
                "active_table": table_name,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return ParseResult.failed(str(e), modality="tabular")


registry.register([".sqlite", ".db"], "tabular", parse_sqlite)