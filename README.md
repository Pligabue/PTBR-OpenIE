# Open Relationship Triple Extraction in Portuguese

## Introduction

The goal of this package is to train models capable of extracting open relationship triples directly from text in Portuguese.

Open relationship triples are triples in the `⟨subject; relation; object⟩` in which every component is a string (usually) taken directly from the text. This means that there is no set of relations or entity types guiding the extraction.

The extraction task is internally split into two sections: predicate extraction (PE) and argument prediction (AP). The first defines the relation, while the second uses the relation to identify the subject and the object.

It is based on Multi²OIE[^1] in terms of the structure of its neural networks, and it uses BERTimbau[^2] to build its word embeddings.

## Usage

### Prerequisites

We recommend that you use Python `3.9` (though `3.10` and `3.11` should also work) and that you have [Poetry](https://python-poetry.org/) installed. 

### Installing

```sh
poetry install
```

### Training a model

Training a model is relatively straightforward, only requiring the use of the main `TripleExtractor` model: 

```py
from triple_extractor_ptbr_pligabue.model import TripleExtractor
```

#### Dataset

To train an extraction model, it is necessary to have a list of annotated sentences. The annotation we used is based on HTML-like tags around each of the three elements in the triple. An annotation for the sentence "Obama was born in Hawaii" would look like this:

```md
<SUBJECT>Obama</SUBJECT> <RELATION>was born in</RELATION> <OBJECT>Hawaii</OBJECT>.
```

If you have a file with one annotated sentence per line, the following should work:

```py
with open("<PATH TO DATASET>", encoding="utf-8") as f:
    training_sentences = f.readlines()
```

#### Model set-up

In terms of customizing the extraction model, you can define the size of the dense layers that are at the end of the PE and the AP, right before the softmax layers. This is done by passing two tuples to the `TripleExtractor` constructor, the first for PE and the second for AP. The length of the tuple defines the number of layers, while the integers in each position define the number of units in each layer. For example, the following model has three layers with 50 units at the end of the PE section and two layers with 25 units at the end of the AP section:

```py
model = TripleExtractor((50, 50, 50), (25, 25))
```

You should then compile your model by running:

```py
model.compile()
```

By default, the compiled model will use `tf.keras.optimizers.SGD` as the optimizer and `tf.keras.losses.CategoricalCrossentropy` to calculate the loss. This can be changed by passing the `pe_optimizer`, `pe_loss`, `ap_optimizer` and `ap_loss` keyword arguments.

Finally, you can check out the model summary by running:

```py
model.summary()
```

#### Fitting the model

You can fit the model by running:

```py
model.fit(training_sentences)
```

At this point, you can change hyperparameters like `batch_size` and `epochs` by passing new values as keyword arguments. For the epochs hyperparameter specifically, you can pass the `pe_epochs` and `ap_epochs` arguments if you want differnt values for each stage.

### Saving and loading

To save a model, all you need to do is pass the desired name:

```py
model.save("<MODEL NAME>")
```

If no name is passed, the name `default` will be used.

Loading a model is also done by name, and the default name is also `default`:

```py
model = TripleExtractor.load("<MODEL NAME>")
```

### Extracting relationship triples

With a trained model, the extraction of relationship triples is simple.

If you wish to annotate sentences and have the result appear on the terminal/notebook, use:

```py
model.annotate_sentences(list_of_sentences)  # list_of_sentences is just a list of strings, each containing one sentence
```

If you wish to process documents and have the results saved in CSV files, use:

```py
model.process_docs(list_of_paths, path_to_csv_dir)  
# list_of_paths is a list of instances of pathlib.Path pointing to TXT files containing the texts.
# path_to_csv_dir is a Path to the directory where the CSV files will be saved.
```

## References

[^1]: https://github.com/youngbin-ro/Multi2OIE
[^2]: https://huggingface.co/neuralmind/bert-base-portuguese-cased