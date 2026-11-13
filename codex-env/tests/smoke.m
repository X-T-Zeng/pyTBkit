H = [1, 1i; -1i, -1];
e = sort(eig(H));
assert(max(abs(e - [-sqrt(2); sqrt(2)])) < 1e-12);
assert(abs(quadgk(@(x) sin(x), 0, pi) - 2) < 1e-10);
assert(max(abs(ifft(fft([1 2 3 4])) - [1 2 3 4])) < 1e-12);
disp('Octave two-band + integration + FFT PASS');
