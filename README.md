# DTris

⚠️ Release binaries are not tested currently and may not work ⚠️

Create vertex color based meshes that have geometric detail around areas where pixel contrast expresses the details.

This is a fun little project where I encounted this great [https://simonschreibt.de/gat/homeworld-2-backgrounds-tech/](article) by Simon Schrebibt and decided to go implement it. 
I used some LLM here and there to to see how it would fair. It largely got many things wrong but also made somethings very quick to impl.

## Build

```
cargo build --release
```

## Usage

```
dtris --help
```

> Example

```
dtris -i nebula_3.png --sample random --detail-samples 3000 --grey-threshold 2 --black-level 16
```

The results will be in the `output` folder named like the input image.  

## Roadmap (sortof)

- [ ] gui interface, see the results live in the application
- [ ] fix GLB output
