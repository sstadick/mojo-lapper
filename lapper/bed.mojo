from ExtraMojo.bstr.bstr import SplitIterator
from ExtraMojo.io.buffered import (
    BufferedReader,
    BufferedWriter,
)
from ExtraMojo.io.delimited import DelimReader, FromDelimited, ToDelimited


@fieldwise_init
struct SimpleBedRecord(Copyable, FromDelimited, Movable, Writable):
    # Cheat with this for now, because Dict seems weirdly slow
    var chr: UInt8
    var start: UInt32
    var stop: UInt32
    var overlaps: UInt32
    var cov: UInt32

    @staticmethod
    fn from_delimited(
        mut data: SplitIterator,
        read header_values: Optional[List[String]] = None,
    ) raises -> Self:
        var chr_str = StringSlice(unsafe_from_utf8=data.__next__())[3:]
        var chr: UInt8
        if chr_str == "X":
            chr = 23
        elif chr_str == "Y":
            chr = 24
        else:
            chr = atol(chr_str)
        # TODO: add my own atol to bstr that can know that it's bytes
        var start = atol(StringSlice(unsafe_from_utf8=data.__next__()))
        var stop = atol(StringSlice(unsafe_from_utf8=data.__next__()))
        return Self(chr, start, stop, 0, 0)

    fn write_to[W: Writer](read self, mut writer: W):
        if self.chr < 23:
            writer.write(
                "chr",
                self.chr,
                "\t",
                self.start,
                "\t",
                self.stop,
                "\t",
                self.overlaps,
                "\t",
                self.cov,
                "\n",
            )
        elif self.chr == 23:
            writer.write(
                "chrX",
                "\t",
                self.start,
                "\t",
                self.stop,
                "\t",
                self.overlaps,
                "\t",
                self.cov,
                "\n",
            )
        else:
            writer.write(
                "chrY",
                "\t",
                self.start,
                "\t",
                self.stop,
                "\t",
                self.overlaps,
                "\t",
                self.cov,
                "\n",
            )

    @staticmethod
    fn read_file(read file: String) raises -> List[Self]:
        var reader = DelimReader[Self](
            BufferedReader(open(String(file), "r")),
            delim=ord("\t"),
            has_header=False,
        )
        return [rec for rec in reader^]


# def main():
#     var file = "/Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed"
#     var records = SimpleBedRecord.read_file(file)
#     print(records[0])
