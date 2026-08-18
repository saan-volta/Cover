
from lib.steganography import *
from lib.mec_math import pad_with_rand, cycle_k

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
    KEY_LENGTH = 11

    assert N_BLOCKS*BLOCK_LENGTH % 8 == 0 # full number of bytes

    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    TOPK = 100

    print(f'\n\n{"-" * 40}\tINITIALIZING\t{"-" * 40} \n\n')

    steg = Steganographer(MODEL_NAME, TOPK, BLOCK_LENGTH)

    # message = input(f'Enter message (max {N_BLOCKS*BLOCK_LENGTH//8} bytes:\t')
    message = 'We will meet tomorrow at sunrise at the old rail tracks'
    message = message.encode() # assume whole number of bytes
    assert 8*len(message) <= BLOCK_LENGTH*N_BLOCKS

    # context = input(f'Enter context (use standard english tokens):\t')
    context = 'Perhaps the weather will hold'

    key = random.getrandbits(KEY_LENGTH)
    cycled_key = cycle_k(key, len(message), KEY_LENGTH)

    print(f'Message:\t{message}\nContext:\t{context}\nKey:\t{key}\n')

    ciphertext = bytes_xor(message, cycled_key)
    cipher_partition: list[bitarray] = pad_with_rand(ciphertext, BLOCK_LENGTH, N_BLOCKS)

    print(f'\n\n{"-"*40}\tENCODING\t{"-"*40} \n\n')

    encodings = steg.encode(cipher_partition, context) # generator
    rtext = Text()
    with Live(rtext, refresh_per_second=10) as live:
        for token, _idx, _deltaH, d_kl in encodings:
            rtext.append(token)
            live.update(Text(f"KL div: \t{d_kl:e}\n\n") + Text(context)+rtext)

    text = rtext.plain

    print(f'\n\n{"-"*40}\tDECODING\t{"-"*40} \n\n')

    decodings = steg.decode(
        steg.llm_sampler.tokenizer.encode(text),
        context,
        N_BLOCKS
    )

    cycled_key_full_length = cycle_k(key, BLOCK_LENGTH*N_BLOCKS//8, KEY_LENGTH)
    with Live(refresh_per_second=10) as live:
        for pred_bitarr_list in decodings:
            out = bytes_xor( b''.join(pred_bitarr_list) , cycled_key_full_length).decode('latin-1')
            live.update(Text(out))



    print(f'\n\n{"-"*40}\tDONE\t{"-"*40} \n\n')


def bytes_xor(B1: bytes, B2: bytes):
    assert len(B1)==len(B2)
    return bytes([b1^b2 for b1,b2 in zip(B1, B2)])


if __name__ == '__main__':
    main()







