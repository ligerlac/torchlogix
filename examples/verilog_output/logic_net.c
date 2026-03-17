#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

void logic_net(long long const *inp, long long *out);

void logic_net(long long const *inp, long long *out) {
	long long linear_buf_a[4];
	long long linear_buf_b[4];

	linear_buf_a[0] = inp[0];
	linear_buf_a[1] = inp[6];
	linear_buf_a[2] = inp[3];
	linear_buf_a[3] = inp[7];
	out[0] = ~0LL;
	out[1] = 0LL;
}