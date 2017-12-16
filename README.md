Self-Organizing Map (SOM) Kohonen artificial neural network
==================

Complete C implementation of the Kohonen artificial neural network algorithm.

Compile
=======

```
gcc kohonen.c -o kohonen
```

Sample usage
=====

```
kohonen manresa_m2_rooms_price.txt
```

The output of this command is a file named 'generated_kohonen_map.html' containing a visual representation of the calculated Kohonen map separated by the components of the input file.

![](http://www.lafruitera.com/kohonen-map-sample.png)

