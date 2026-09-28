#!/usr/bin/env python
"""Minimalny przykład ciągłego pomiaru prądu metodą IIR albo Scope DMA.

Tryb IIR odczytuje pojedynczą, przefiltrowaną wartość bezpośrednio z układu
FPGA. Tryb DMA odbiera kompletne ramki przebiegu z oscyloskopu i przekazuje
do dalszego użycia wyłącznie najnowszą z nich. Oba pomiary działają w pętli
bez sztucznego opóźnienia i kończą się po naciśnięciu Ctrl+C.

Otrzymywane liczby są surowymi kodami związanymi z przetwornikiem ADC, a nie
wartościami w amperach. Przeliczenie na prąd wymaga kalibracji odpowiedniej
dla użytego czujnika oraz analogowego toru wejściowego. Opcjonalny callback
może wyświetlać, zapisywać lub przekazywać najnowsze dane do innego wątku.

Callback to zwykła funkcja przekazana jako argument innej funkcji. Kod
pomiarowy wywołuje ją automatycznie po uzyskaniu nowych danych. W tym
przykładzie callback przyjmuje dokładnie jeden argument: liczbę ``int`` dla
IIR albo tablicę próbek ``numpy.ndarray`` dla DMA. Wynik zwracany przez
callback jest ignorowany. Callback może na przykład:

* wypisać bieżącą wartość w terminalu,
* przeliczyć surowe kody na ampery,
* zapisać dane do pliku lub bazy danych,
* zaktualizować wykres,
* umieścić dane w kolejce obsługiwanej przez inny wątek.

W ``main()`` callbacki są krótkimi funkcjami ``lambda``, które tylko drukują
wynik. Można je zastąpić własną funkcją. Wyjątek zgłoszony wewnątrz callbacku
przerywa pomiar, ale sekcja ``finally`` nadal zamyka urządzenie.

Ustaw ``USE_DMA = True``, aby uruchomić pomiar DMA. Wartość ``False`` wybiera
prostszy odczyt IIR.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from nlab.hardware.digitizer import Digitizer
from nlab.hardware.digitizer.dma import IIOScopeDmaStreamer, ScopeFrameBuffer
from nlab.hardware.digitizer.scope import TriggerMode

URI = "ip:192.168.10.128:30431"
CHANNEL = 0
USE_DMA = True


def measure_current_iir(
    digitizer: Digitizer,
    callback: Callable[[int], None] | None = None,
) -> None:
    """Odczytuje kolejne wartości prądu z filtra IIR aż do Ctrl+C.

    Każdy obrót pętli wykonuje jeden bezpośredni odczyt z FPGA. Nie ma tutaj
    wywołania ``sleep()``, dlatego częstotliwość próbkowania ogranicza jedynie
    czas komunikacji z urządzeniem oraz czas działania callbacku. Jeżeli
    ``callback`` nie jest ``None``, otrzymuje najnowszy surowy kod jako ``int``.
    Callback działa w tej samej pętli, dlatego długie obliczenia, zapis na
    wolny dysk lub komunikacja sieciowa zmniejszą częstotliwość odczytów IIR.
    W takim przypadku callback powinien jedynie szybko przekazać wartość do
    kolejki, a właściwe przetwarzanie należy wykonać w osobnym wątku.
    """
    while True:
        value = int(digitizer.mca.filters.lp.get_iir_average())
        if callback is not None:
            # Przy kosztownym przetwarzaniu można tutaj wstawić ``value`` do
            # kolejki, a kolejkę obsługiwać w osobnym wątku roboczym. Dzięki
            # temu callback nie będzie opóźniał następnego odczytu z FPGA.
            callback(value)


def measure_current_dma(
    digitizer: Digitizer,
    callback: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Odbiera ramki Scope DMA aż do Ctrl+C i przekazuje najnowszą ramkę.

    Oscyloskop jest ustawiany w okresowym trybie wyzwalania. Osobny wątek
    nieprzerwanie odbiera kompletne ramki DMA, natomiast bieżący wątek pobiera
    najnowszą dostępną ramkę z bufora o rozmiarze jeden. Jeżeli konsument jest
    wolniejszy od DMA, starsze ramki podglądu są zastępowane, ale sam odbiór
    danych nie jest blokowany. Callback otrzymuje tablicę próbek ``int16``.
    Może na przykład obliczyć średnią ramki, zastosować kalibrację, zbudować
    wykres albo przekazać ramkę do dalszej analizy. Nie powinien zakładać, że
    zobaczy każdą ramkę: bufor celowo zachowuje wyłącznie najnowszą dostępną.

    Blok ``finally`` zatrzymuje oscyloskop, anuluje ewentualny blokujący odczyt
    i czeka na zakończenie wątku DMA przed zamknięciem połączenia.
    """
    streamer = digitizer.scope_dma
    if not isinstance(streamer, IIOScopeDmaStreamer):
        raise RuntimeError("Scope DMA requires the direct IIO backend")

    scope = digitizer.scope
    latest_only = ScopeFrameBuffer(max_frames=1)
    stop_event = threading.Event()
    frame_ready = threading.Event()
    errors: list[str] = []

    def capture() -> None:
        """Odbiera pełne ramki DMA w osobnym wątku pomiarowym."""
        try:
            streamer.stream_to_file(
                None,
                stop_event,
                on_progress=lambda _bytes: frame_ready.set(),
                frame_buffer=latest_only,
            )
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            frame_ready.set()

    worker: threading.Thread | None = None
    try:
        scope.stop()
        scope.set_pretrigger_samples(0)
        scope.set_frame_samples(8188)
        scope.set_trigger_mode(TriggerMode.TIMED)
        scope.set_frame_period_cycles(12_500)

        worker = threading.Thread(target=capture, name="current-dma")
        worker.start()
        while worker.is_alive():
            frame_ready.wait()
            frame_ready.clear()
            frames, _ = latest_only.drain()
            if frames and callback is not None:
                # Wątek ``worker`` nadal odbiera DMA, gdy ten wątek przetwarza
                # najnowszą ramkę. Wolniejszy callback nie zatrzyma więc DMA.
                callback(np.asarray(frames[-1].samples, dtype=np.int16))
            if errors:
                raise RuntimeError(f"Scope DMA failed: {errors[0]}")
    finally:
        try:
            scope.stop()
        finally:
            stop_event.set()
            if worker is not None:
                try:
                    streamer.request_stop()
                finally:
                    worker.join()


def main() -> None:
    """Łączy się z urządzeniem i uruchamia wybrany tryb aż do Ctrl+C."""
    digitizer = Digitizer.from_iio(CHANNEL, URI, with_ids=False)
    try:
        if USE_DMA:
            measure_current_dma(
                digitizer,
                callback=lambda frame: print(
                    f"DMA mean raw current: {frame.mean():.1f}"
                ),
            )
        else:
            measure_current_iir(
                digitizer,
                callback=lambda value: print(f"IIR raw current: {value}"),
            )
    except KeyboardInterrupt:
        print("\nMeasurement stopped.")
    finally:
        digitizer.close()


if __name__ == "__main__":
    main()
