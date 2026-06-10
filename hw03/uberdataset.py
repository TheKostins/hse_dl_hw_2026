import pandas as pd
from dataclasses import dataclass
from typing import Callable, Any, Optional, Iterable
import hashlib


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    deps: tuple[str, ...]
    produces: tuple[str, ...]


type FeatureFunc = Callable[[pd.DataFrame], Optional[pd.DataFrame]]
type RegistryEntry = tuple[FeatureSpec, FeatureFunc]
type Registry  = dict[str, RegistryEntry]


def _fn_hash(fn: Callable) -> str:
    h = hashlib.sha256()
    h.update(fn.__code__.co_code)
    h.update(str(fn.__code__.co_consts).encode())
    return h.hexdigest()


def _topsort(registry: Registry, targets: tuple[str, ...]) -> list[str]:
    """Return topological order for targets and their transitive dependencies."""
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(name: str):
        if name not in registry:
            raise KeyError(f"Unknown feature '{name}'. Register it before building.")
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Cyclic dependency detected at {name}")
        visiting.add(name)
        spec, _ = registry[name]
        for dep in spec.deps:
            dfs(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for target in targets:
        dfs(target)

    return order


class UberDatasetFuhrer:
    """
    Feature orchestration wrapper around a Pandas DataFrame.

    Features are registered as callables that mutate `self.df` in place (or
    return a replacement DataFrame) and can declare dependencies on other
    features. `build()` resolves dependencies and executes features in a valid
    order with simple staleness tracking.

    Optionally, `feature_fn_prefix` can be configured to strip a shared prefix
    from function names when `@feature` is used without an explicit `name`.
    Example: with `feature_fn_prefix="f_"`, `def f_price(...)` is registered
    as feature `"price"`.
    """

    def __init__(self, df: pd.DataFrame, *, feature_fn_prefix: str | None = None):
        """
        Args:
            df: Backing dataframe that features read and mutate.
            feature_fn_prefix: Optional prefix removed from implicit feature
                names derived from function names.
        """
        if feature_fn_prefix == "":
            raise ValueError("feature_fn_prefix must be None or a non-empty string")
        if feature_fn_prefix is not None and not isinstance(feature_fn_prefix, str):
            raise TypeError("feature_fn_prefix must be None or str")

        self.df = df
        self._feature_fn_prefix = feature_fn_prefix
        self._registry: Registry = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def _resolve_feature_name(self, raw_name: str, *, explicit: bool) -> str:
        """Resolve canonical registry name from input/derived feature name."""
        if explicit:
            return raw_name

        prefix = self._feature_fn_prefix
        if prefix and raw_name.startswith(prefix):
            resolved = raw_name[len(prefix):]
            if not resolved:
                raise ValueError(
                    f"Feature name '{raw_name}' becomes empty after stripping prefix '{prefix}'"
                )
            return resolved

        return raw_name

    def feature(
        self,
        name: Optional[str] = None,
        *,
        deps: Optional[Iterable[str]] = None,
        produces: Optional[Iterable[str]] = None,
    ):
        """
        Decorator for registering a feature function.

        Args:
            name: Feature identifier. Defaults to function name.
                If omitted, constructor-level `feature_fn_prefix` stripping
                is applied when the function name starts with that prefix.
            deps: Upstream feature names that must be built first.
            produces: DataFrame column names produced by this feature. Defaults
                to a single column named as the feature.
        """

        deps_t = tuple(deps or ())
        produces_t = tuple(produces) if produces is not None else None

        def decorator(fn: FeatureFunc):
            base_name = name if name is not None else fn.__name__
            feature_name = self._resolve_feature_name(base_name, explicit=(name is not None))
            final_produces = produces_t if produces_t is not None else (feature_name,)

            spec = FeatureSpec(
                name=feature_name,
                deps=deps_t,
                produces=final_produces,
            )
            self._registry[spec.name] = (spec, fn)
            return fn

        return decorator

    def is_fresh(self, name: str) -> bool:
        """
        Return whether feature code hash and declared outputs are still valid.

        Args:
            name: Feature name.

        Returns:
            `True` if metadata exists, function hash matches, and all declared
            output columns are present in `self.df`.
        """
        spec, fn = self._registry[name]
        meta = self._meta.get(name)
        if meta is None:
            return False
        if meta["hash"] != _fn_hash(fn):
            return False
        return all(col in self.df.columns for col in spec.produces)

    def register(
        self,
        fn: FeatureFunc,
        *,
        name: Optional[str] = None,
        deps: Optional[Iterable[str]] = None,
        produces: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Register a feature function programmatically.

        Args:
            fn: Feature function.
            name: Feature identifier. Defaults to function name.
            deps: Upstream feature names.
            produces: Declared output columns. Defaults to `(name,)`.
        """
        _ = self.feature(name=name, deps=deps, produces=produces)(fn)

    def build(self, *targets: str, force: bool = False, drop_na: bool = True) -> "UberDatasetFuhrer":
        """
        Build target features and all transitive dependencies.

        Args:
            targets: Feature names to build.
            force: If `True`, rebuild every feature in dependency order.
            drop_na: If `True`, drop rows with missing values after building features.
        """
        order = _topsort(self._registry, targets)
        rebuilt: set[str] = set()

        for name in order:
            spec, fn = self._registry[name]
            dependency_rebuilt = any(dep in rebuilt for dep in spec.deps)

            if (not force) and (not dependency_rebuilt) and self.is_fresh(name):
                continue

            result = fn(self.df)
            if result is not None:
                if not isinstance(result, pd.DataFrame):
                    raise TypeError(
                        f"Feature '{name}' must return None or pandas.DataFrame, got {type(result)!r}"
                    )
                self.df = result

            missing = [c for c in spec.produces if c not in self.df.columns]
            if missing:
                raise RuntimeError(
                    f"Feature {name} is supposed to produce {spec.produces}, but missing {missing} in dataframe"
                )

            self._meta[name] = {"hash": _fn_hash(fn)}
            rebuilt.add(name)

        if drop_na:
            self.df.dropna(inplace=True)

        return self

    def build_all(self, *, force: bool = False, drop_na: bool = True):
        """Build every registered feature in dependency-resolved order."""
        names = sorted(self._registry.keys())
        return self.build(*names, force=force, drop_na=drop_na)

    def make_splits(
        self,
        *,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
        shuffle: bool = True,
        key: str = "_split",
    ) -> "UberDatasetFuhrer":
        """
        Assign deterministic train/val/test split labels to rows.

        Args:
            train: Fraction of rows assigned to the training split.
            val: Fraction of rows assigned to the validation split.
            test: Fraction of rows assigned to the test split.
            seed: RNG seed used when `shuffle=True`.
            shuffle: If `True`, shuffle row indices before splitting.
            key: Name of output column containing split labels.

        Returns:
            `self` for fluent chaining.
        """
        if abs((train + val + test) - 1.0) > 1e-9:
            raise ValueError("train+val+test must sum to 1.0")

        n = len(self.df)
        idx = self.df.index.to_numpy()

        if shuffle:
            import numpy as np
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        self.splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        self.df[key] = "test"
        self.df.loc[train_idx, key] = "train"
        self.df.loc[val_idx, key] = "val"

        return self
