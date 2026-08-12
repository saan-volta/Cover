
from lib.steganography import *
from lib.mec_math import pad_with_rand

import random
from functools import reduce
from bitarray import bitarray
from bitarray.util import int2ba, ba2int

# demo
from rich.live import Live
from rich.text import Text


def main():
    '''
    Usage example
    '''

    # set up
    N_BLOCKS = 64
    BLOCK_LENGTH = 8 # bits
    KEY_LENGTH = BLOCK_LENGTH # for simplicity

    # Note: 8 = KEY_LENGTH = BLOCK_LENGTH

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    TOPK = 40

    print(f'\n\n{"-" * 40}\tINITIALIZING\t{"-" * 40} \n\n')

    steg = Steganographer(MODEL_NAME, TOPK, BLOCK_LENGTH)

    message = 'Hello world. This is an example encoding procedure.'.encode() # assume whole number of bytes
    assert 8*len(message) <= BLOCK_LENGTH*N_BLOCKS

    context = 'The weather' # ...

    key = int2ba( random.randint(0, (1<<KEY_LENGTH) - 1), KEY_LENGTH)

    print(f'Message:\t{message}\nContext:\t{context}\nKey:\t{ba2int(key)}\n')

    message_partition: list[bitarray] = pad_with_rand(message, BLOCK_LENGTH, N_BLOCKS)
    # print(len(key), len(message_partition[0]))
    cipher_partition: list[bitarray] = [mp ^ key for mp in message_partition]

    print(f'\n\n{"-"*40}\tENCODING\t{"-"*40} \n\n')

    encodings = steg.encode(cipher_partition, context) # generator
    rtext = Text()
    with Live(rtext, refresh_per_second=10) as live:
        for token, _1, _2 in encodings:
            rtext.append(token)
            live.update(Text(context)+rtext)

    text = rtext.plain

    print(f'\n\n{"-"*40}\tDECODING\t{"-"*40} \n\n')

    decodings = steg.decode(
        steg.llm_sampler.tokenizer.encode(text),
        context,
        N_BLOCKS
    )

    with Live(refresh_per_second=10) as live:
        for pred_bitarr_list in decodings:
            out = b''.join([
                (barr ^ key) for barr in pred_bitarr_list
            ]).decode('latin-1')
            live.update(Text(out))



    print(f'\n\n{"-"*40}\tDONE\t{"-"*40} \n\n')


def bytes_xor(B1: bytes, B2: bytes):
    assert len(B1)==len(B2)
    return bytes([b1^b2 for b1,b2 in zip(B1, B2)])


if __name__ == '__main__':
    main()







