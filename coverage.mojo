from sys import stdout, stderr
from time import perf_counter

from ExtraMojo.cli.parser import OptParser, OptConfig, OptKind
from ExtraMojo.io.buffered import BufferedReader, BufferedWriter
from ExtraMojo.io.delimited import DelimReader

from lapper.lapper import Lapper, Interval
from lapper.bed import SimpleBedRecord


def main():
    var parser = OptParser(
        name="coverage", description="Compute the coverage of BED A vs BED B."
    )
    parser.add_opt(
        OptConfig(
            "bed-a",
            OptKind.StringLike,
            default_value=None,
            description="bedfile A",
        )
    )
    parser.add_opt(
        OptConfig(
            "bed-b",
            OptKind.StringLike,
            default_value=None,
            description="bedfile B",
        )
    )
    parser.add_opt(
        OptConfig(
            "method",
            OptKind.StringLike,
            default_value=String("lapper_find"),
            description=(
                "Which bench method to run: [lapper_find,"
                " lapper_find_vectorized, ezlapper_find"
            ),
        )
    )
    var opts = parser.parse_sys_args()
    var bed_a = opts.get_string("bed-a")
    var bed_b = opts.get_string("bed-b")
    var method = opts.get_string("method")

    if method == "lapper_find":
        lapper_find(bed_a, bed_b)
    elif method == "lapper_find_vectorized":
        lapper_find_vectorized(bed_a, bed_b)
    else:
        print("Unsupported method:", method, file=stderr)


# TODO: fix the mismatch in storage or whatever is hapening
# TODO: try dict version again in case I was out of bounds and that was the slow
# Add vectorized version
# Add gpu version(s)
# Add own dict version


fn lapper_find(bed_a: String, bed_b: String) raises:
    var start_setup = perf_counter()
    var reader = DelimReader[SimpleBedRecord](
        BufferedReader(open(String(bed_a), "r")),
        delim=ord("\t"),
        has_header=False,
    )

    # Cheating with no dict, but dict is weirdly slow
    var bed_ivs: List[List[Interval]] = [List[Interval]() for _ in range(0, 24)]
    var i = 0
    for rec in reader^:
        bed_ivs[rec.chr - 1].append(Interval(rec.start, rec.stop, i))
    print("Creating lappers", file=stderr)
    # var bed = Dict[String, Lapper]()
    var bed = List[Optional[Lapper]](unsafe_uninit_length=len(bed_ivs))
    while len(bed_ivs) > 0:
        var i = len(bed_ivs) - 1

        var x = Pointer(to=bed_ivs.pop(i))
        if len(x[]) > 0:
            bed[i] = Lapper(x[])
        else:
            bed[i] = None
    var end_setup = perf_counter()

    print("Setup time:", end_setup - start_setup, file=stderr)

    var readerB = DelimReader[SimpleBedRecord](
        BufferedReader(open(String(bed_b), "r")),
        delim=ord("\t"),
        has_header=False,
    )
    var writer = BufferedWriter(stdout)

    var found = List[Interval]()
    for rec in readerB^:
        if not bed[rec.chr - 1]:
            # if rec.chr not in bed:
            writer.write(rec)  # cov is 0 by defulat
            continue

        var cov_start: UInt32 = 0
        var cov_stop: UInt32 = 0
        ref lapper = bed[rec.chr - 1].value()
        found.clear()
        lapper.find(rec.start, rec.stop, found)
        rec.overlaps += len(found)
        for iv in found:
            var start = iv.start if iv.start > rec.start else rec.start
            var stop = iv.stop if iv.stop < rec.stop else rec.stop
            if start > cov_stop:
                rec.cov += cov_stop - cov_start
                cov_start = start
                cov_stop = stop
            else:
                cov_stop = stop if cov_stop < stop else cov_stop
        rec.cov += cov_stop - cov_start
        writer.write(rec)


fn lapper_find_vectorized(bed_a: String, bed_b: String) raises:
    var start_setup = perf_counter()
    var reader = DelimReader[SimpleBedRecord](
        BufferedReader(open(String(bed_a), "r")),
        delim=ord("\t"),
        has_header=False,
    )

    # Cheating with no dict, but dict is weirdly slow
    var bed_ivs: List[List[Interval]] = [List[Interval]() for _ in range(0, 24)]
    var i = 0
    for rec in reader^:
        bed_ivs[rec.chr - 1].append(Interval(rec.start, rec.stop, i))
    print("Creating lappers", file=stderr)
    # var bed = Dict[String, Lapper]()
    var bed = List[Optional[Lapper]](unsafe_uninit_length=len(bed_ivs))
    while len(bed_ivs) > 0:
        var i = len(bed_ivs) - 1

        var x = Pointer(to=bed_ivs.pop(i))
        if len(x[]) > 0:
            bed[i] = Lapper(x[])
        else:
            bed[i] = None
    var end_setup = perf_counter()

    print("Setup time:", end_setup - start_setup, file=stderr)

    var readerB = DelimReader[SimpleBedRecord](
        BufferedReader(open(String(bed_b), "r")),
        delim=ord("\t"),
        has_header=False,
    )
    var writer = BufferedWriter(stdout)

    var found = List[UInt32]()
    for rec in readerB^:
        if not bed[rec.chr - 1]:
            # if rec.chr not in bed:
            writer.write(rec)  # cov is 0 by defulat
            continue

        var cov_start: UInt32 = 0
        var cov_stop: UInt32 = 0
        ref lapper = bed[rec.chr - 1].value()
        found.clear()
        lapper.find_vectorized(rec.start, rec.stop, found)
        rec.overlaps += len(found)
        for iv_idx in found:
            var iv_start = lapper.starts[iv_idx]
            var iv_stop = lapper.stops[iv_idx]
            var start = iv_start if iv_start > rec.start else rec.start
            var stop = iv_stop if iv_stop < rec.stop else rec.stop
            if start > cov_stop:
                rec.cov += cov_stop - cov_start
                cov_start = start
                cov_stop = stop
            else:
                cov_stop = stop if cov_stop < stop else cov_stop
        rec.cov += cov_stop - cov_start
        writer.write(rec)


# The dict way of doing it that is weirdly slow
# fn lapper_find_vectorized(bed_a: String, bed_b: String) raises:
#     pass
# var start_setup = perf_counter()
# var reader = DelimReader[SimpleBedRecord](
#     BufferedReader(open(String(bed_a), "r")),
#     delim=ord("\t"),
#     has_header=False,
# )

# var bed_ivs = Dict[String, List[Interval]]()
# var i = 0
# for rec in reader^:
#     if rec.chr not in bed_ivs:
#         bed_ivs[rec.chr] = List[Interval](Interval(rec.start, rec.stop, i))
#     else:
#         bed_ivs[rec.chr].append(Interval(rec.start, rec.stop, i))

# var bed = Dict[String, Lapper]()
# while len(bed_ivs) > 0:
#     var entry = bed_ivs.popitem()  # TODO verify no copy here
#     bed[entry.key] = Lapper(entry.reap_value())
# var end_setup = perf_counter()
# print("Setup time:", end_setup - start_setup, file=stderr)

# var readerB = DelimReader[SimpleBedRecord](
#     BufferedReader(open(String(bed_b), "r")),
#     delim=ord("\t"),
#     has_header=False,
# )
# var writer = BufferedWriter(stdout)

# var found = List[UInt32]()
# for rec in readerB^:
#     if rec.chr not in bed:
#         writer.write(rec)  # cov is 0 by defulat
#         continue

#     var cov_start: UInt32 = 0
#     var cov_stop: UInt32 = 0
#     ref lapper = bed[rec.chr]
#     found.clear()
#     lapper.find_vectorized(rec.start, rec.stop, found)
#     rec.overlaps += len(found)
#     for iv_idx in found:
#         var iv_start = lapper.starts[iv_idx]
#         var iv_stop = lapper.stops[iv_idx]
#         var start = iv_start if iv_start > rec.start else rec.start
#         var stop = iv_stop if iv_stop < rec.stop else rec.stop
#         if start > cov_stop:
#             rec.cov += cov_stop - cov_start
#             cov_start = start
#             cov_stop = stop
#         else:
#             cov_stop = stop if cov_stop < stop else cov_stop
#     rec.cov += cov_stop - cov_start
#     writer.write(rec)
