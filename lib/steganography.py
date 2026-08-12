
from .mec_math import *
from .llm_sampler import *

from collections.abc import Generator

class Steganographer:
    def __init__(self, model_name, topk, block_size):
        self.llm_sampler = Sampler(model_name, topk) # autoregressive conditional distribution

        self.topk = topk
        self.block_size = block_size


    def encode(self, ct_blocks: list[bitarray], context: str, max_iters=1000) -> Generator[str, None, None]:
        '''
        Args:
            ct_blocks: list of bitarrays each of width BLOCK_SIZE padded appropriately.
            context: covertext string
            max_iters: hard cap on number of steps. May fail to embed full information of the input

        Returns:
            Generator of strings (tokens)
        '''

        # ciphertext_bits_arr = bitarray(ciphertext.encode())

        mu_num = 1<< self.block_size
        # ct_blocks = block_partition(ciphertext_bits_arr, self.block_size)
        ct_blocks_idxs = list(map(barr_to_int, ct_blocks))
        mus = [np.ones(mu_num, ) / mu_num for _ in ct_blocks]  # init uniforms
        mus_entropy = np.array([entropy(mu) for mu in mus])

        # autoregressive conditional: p(C_j | C_1:j-1 = S_1:j-1)
        C_probs, C_idxs = self.llm_sampler.start(context)
        C_probs = normalize(C_probs)
        S = []

        j = 0
        while mus_entropy.max() > 0 and j < max_iters:

            istar = np.argmax(mus_entropy)  # i in [0, len(mus)]
            mu_istar = mus[istar]
            # coupling
            M = mec(mu_istar, C_probs)  # (mu_dom, topk)
            # next token conditional on block value
            d_token = normalize(M[ct_blocks_idxs[istar]])  # (topk, 1)
            S_j_ix = np.random.choice(np.arange(0, self.topk), p=d_token)  # S_j_ix is index in [0, topk]
            S_j = C_idxs[S_j_ix]  # S_j is token id in [0, VOCAB_SIZE]
            S.append(S_j)
            # update context, generate new AC distribution
            # context += self.tokenizer.decode([S_j])
            C_probs, C_idxs = self.llm_sampler.step(S_j)
            C_probs = normalize(C_probs)

            # condition on realization of next token
            mu_istar_prime = normalize(M[:, S_j_ix])
            # update
            mus[istar] = mu_istar_prime
            delta_entropy = mus_entropy[istar] - entropy(mu_istar_prime)
            mus_entropy[istar] = entropy(mu_istar_prime)

            j += 1
            yield self.llm_sampler.tokenizer.decode([S_j]), istar, delta_entropy

            # print(step)
        # return S, self.llm_sampler.tokenizer.decode(S)


    def decode(self, S: list[int], context, n_blocks) -> Generator[list[bitarray], None, None]:
        """
        Args:
            S:  list of tokens encoded to indices by the LLM's tokenizer. Does not include context.
            context: must be same as encode's
            n_blocks: length in blocks of the encoded message (must be same as encode's)

        Returns:
            Generator yielding list of bitarrays of the predicted ciphertext at each step.
        """

        def sample_from_mu_prod():
            out_barr_list = []
            block_ids = []
            for mu in mus:
                idcand = np.random.choice(np.arange(1 << self.block_size), p=mu)
                block_ids.append(idcand)
                barr = int_to_barr(idcand, width=self.block_size)
                out_barr_list.append(barr)
            return out_barr_list

        mu_num = 1 << self.block_size
        mus = [np.ones(mu_num, ) / mu_num for _ in range(n_blocks)]  # init uniforms
        mus_entropy = np.array([entropy(mu) for mu in mus])
        C_probs, C_idxs = self.llm_sampler.start(context)
        C_probs = normalize(C_probs)

        j = 0
        while mus_entropy.max() > 0 and j < len(S):
            istar = np.argmax(mus_entropy)  # i in [0, len(mus)]
            mu_istar = mus[istar]
            # coupling
            M = mec(mu_istar, C_probs)  # (mu_dom, topk)

            s_j_ix = np.where(C_idxs == S[j])[0].item()  # find column index matching token value

            # update context, generate new conditional distribution
            # context += self.tokenizer.decode([S[step]])
            C_probs, C_idxs = self.llm_sampler.step(S[j])
            C_probs = normalize(C_probs)

            # condition on realization of next token
            mu_istar_prime = normalize(M[:, s_j_ix])
            # update
            mus[istar] = mu_istar_prime
            mus_entropy[istar] = entropy(mu_istar_prime)

            j += 1
            yield sample_from_mu_prod()




