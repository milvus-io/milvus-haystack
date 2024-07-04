# Contributing to Haystack

## Contribute code

### Run code quality checks locally```

Install and update your [ruff](https://github.com/astral-sh/ruff) and [hatch](https://github.com/pypa/hatch) to the latest versions.

To check your code style according to linting rules run:
```sh
hatch run lint:all
````

If the linters spot any error, you can fix it before checking in your code:
```sh
hatch run lint:fmt
```
