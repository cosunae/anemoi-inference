# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io
import logging
from typing import Any
from typing import Literal

import earthkit.data as ekd

from anemoi.inference.context import Context
from anemoi.inference.grib.encoding import encode_message
from anemoi.inference.metadata import Metadata
from anemoi.inference.outputs.grib import BaseGribOutput
from anemoi.inference.types import FloatArray
from anemoi.inference.types import ProcessorConfig

from . import output_registry

LOG = logging.getLogger(__name__)


@output_registry.register("fdb")
class FDBOutput(BaseGribOutput):
    """Write inference output directly to FDB (Fields DataBase) via pyfdb.

    Each GRIB message is encoded in memory and archived to FDB immediately,
    avoiding intermediate GRIB files on disk.

    Example inference config::

        output:
          fdb:
            encoding:
              typeOfGeneratingProcess: 2
              centre: lssw
            templates:
              samples: resources/templates_index_icon.yaml
            post_processors:
              - extract_mask:
                  mask: "lam_0/cutout_mask"
                  as_slice: true

    To target a non-default FDB instance pass ``fdb_config`` and/or
    ``fdb_userconfig`` with the same structure as the FDB5 YAML config.
    """

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        fdb_config: dict | None = None,
        fdb_userconfig: dict | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        encoding: dict[str, Any] | None = None,
        templates: list[str] | str | None = None,
        grib1_keys: dict[str, Any] | None = None,
        grib2_keys: dict[str, Any] | None = None,
        modifiers: list[str] | None = None,
        variables: list[str] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        negative_step_mode: Literal["error", "write", "skip"] = "error",
    ) -> None:
        super().__init__(
            context,
            metadata,
            post_processors=post_processors,
            encoding=encoding,
            templates=templates,
            grib1_keys=grib1_keys,
            grib2_keys=grib2_keys,
            modifiers=modifiers,
            variables=variables,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
            negative_step_mode=negative_step_mode,
        )
        self._fdb_config = fdb_config
        self._fdb_userconfig = fdb_userconfig
        self._fdb = None  # lazy — so construction is safe without an FDB5 library

    @property
    def fdb(self):
        if self._fdb is None:
            try:
                import pyfdb
            except ImportError as exc:
                raise ImportError(
                    "pyfdb is required for FDB output. Install it with: pip install pyfdb"
                ) from exc

            kwargs = {}
            if self._fdb_config is not None:
                kwargs["config"] = self._fdb_config
            if self._fdb_userconfig is not None:
                kwargs["userconfig"] = self._fdb_userconfig

            self._fdb = pyfdb.FDB(**kwargs)
            LOG.info("Opened FDB connection")

        return self._fdb

    def write_message(self, message: FloatArray, template: ekd.Field, **keys: Any) -> None:
        """Encode *message* as a GRIB field and archive it to FDB."""

        # Mirror the defensive guard from GribFileOutput: hide the "name" metadata
        # attribute on the template so eccodes does not trip over it during encoding.
        if template is not None and template.metadata("name", default=None) is not None:

            class _TemplateProxy:
                """Thin proxy that exposes only the eccodes handle."""

                def __init__(self, t: ekd.Field) -> None:
                    self.handle = t.handle

                def __repr__(self) -> str:
                    return f"_TemplateProxy({template!r})"

            template = _TemplateProxy(template)

        try:
            handle = encode_message(
                values=message,
                template=template,
                metadata=keys,
                check_nans=self.context.allow_nans,
            )
        except Exception:
            LOG.error("Error encoding GRIB message for FDB, keys=%s", keys)
            raise

        buf = io.BytesIO()
        handle.write(buf)
        data = buf.getvalue()

        if not data:
            LOG.warning("Empty GRIB message for keys=%s — skipping FDB archive", keys)
            return

        LOG.debug(
            "Archiving %d bytes to FDB (param=%s date=%s step=%s)",
            len(data),
            keys.get("param"),
            keys.get("date"),
            keys.get("step"),
        )
        self.fdb.archive(data)

    def close(self) -> None:
        """Flush and release the FDB connection."""
        if self._fdb is not None:
            LOG.info("Flushing FDB")
            self._fdb.flush()
            self._fdb = None

    def __repr__(self) -> str:
        return "FDBOutput()"
