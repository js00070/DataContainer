# DataContainer

"More than an ECS, less than a database"

## What is it?

The DataContainer project contains a code generation tool (DataContainerGenerator) that turns text specifications describing objects and their relationships into a C++ header files defining a `data_container` class that manages the described objects and relationships.

## Why does this exist?

Since that sounds a lot like what structs and classes in a C++ header file already are, why would anyone want to go through the extra trouble required to use a code generator? The generated data container is intended to solve the following problems:

- Values stored in standard C++ objects end up scattered in memory in a way that is relatively unfriendly to SIMD operations. The standard C++ solution is, on paper, to use a "struct of arrays" rather than an "array of structs", but actually doing this can be quite cumbersome. The generated data container, however, automatically stores values for the same property in contiguous memory and exposes a user-friendly SIMD interface to them.
- Managing relationships between objects involves both repetitive boilerplate and is a common source of bugs. The code required to manage the relationships is also often hard to make changes to without significant effort.[^1] The generated data container helps solve this problem by automatically generating the code required to keep relationships up to date as objects are created and deleted, and automatically maintains any indexes you require for quickly finding an object based on a given relationship, or vice versa.
- Serializing and deserializing C++ objects is a pain in the neck. It is easy enough to dump their contents out to a file, but anything non trivial requires more than that. You may want the ability to selectively save some of an object's properties and not others when it is possible to recalculate that information without saving it directly. And you almost certainly want the ability to load serialized data created by old versions of your software, even after member variables have been added, removed, or changed type. Because the generator has a description of your objects and relationships that it can understand as a whole, it can provide you with robust serialization and deserialization routines without any additional effort on your part.
- And yes, there are existing libraries for doing all of the above things (often quite good ones). However, the problems discussed above are all fundamentally about storing and managing the data that is structured into objects and relationships. Thus, using a library to solve one of these problems may very well make it harder to use a library to solve another of them.[^2] And so the approach of this project is to try to solve them all in one go, and thus to be able to hide the sometimes ugly details from the end user completely.
- OK, but why a code generator? Some early iterations of these ideas were implemented with dark template magics, but doing so made compilation significantly slower. So while it is harder to initially integrate a code generator into your project, I think that ultimately it is the more user-friendly solution.

## Disclaimer

Every effort will be made to find and eliminate bugs, and to keep them eliminated with tests. However, at this stage there are insufficient tests, and insufficient usage experience, to reasonably expect the generated files to be 100% bug free. DO NOT USE A GENERATED DATA CONTAINER IN SOFTWARE WHERE LIVES OR LIVELIHOODS MAY DEPEND ON THE CORRECT OPERATION.

## Documentation

The following documentation explains all you need to know about how to generate and use data containers.

- [Getting started](getting_started.md)
- [Overview](overview.md)
- [Objects and properties](objects_and_properties.md)
- [Relationships](relationships.md)
- [Queries](queries.md)
- [Serialization and deserialization](serialization.md)
- [Multithreading](multithreading.md)
- [The ve SIMD library](ve_documentation.md)
- [File format](file_format_documentation.md)

## Examples and Tutorials

- [Tutorial: integrating a color struct](tutorial_color.md)

## Other

- [Change log](changes.md)

## Feedback

Feel free to leave comments and suggestions as you see fit. If you are actively using, or intend to use, a data container generated by this project, I will try to prioritize any fixes or features that you need. Otherwise, additions will be mostly driven by my own needs.

[^1]: This is the primary reason that this version the idea exists (as an improvement to a previous version that generated containers for each logical object individually); managing the code required to keep the relationships working correctly, and to make sure that they serialized and deserialized correctly, was tedious and error prone in a way that made me unwilling to make changes at all.

[^2]: For example, a robust serialization library is typically build out of techniques that emulate compile-time reflection on C++ classes and structs. This tends to be hard to integrate with the techniques you might use to generate SIMD-friendly memory layouts.