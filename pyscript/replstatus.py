import asyncio
from pyscript import Plugin


class ReplStatus(Plugin):
    def init(self, *args):
        ...

    async def beforePyReplExec(self, *, interpreter, src, outEl, pyReplTag):
        await self.setRunButtonState(pyReplTag, running=True)

    async def afterPyReplExec(self, *, interpreter, src, outEl, pyReplTag, result):
        await self.setRunButtonState(pyReplTag, running=False)

    async def setRunButtonState(self, pyReplTag, running):
        button = pyReplTag.querySelector('.py-repl-run-button')
        svg = button.querySelector('svg')

        if running:
            button.style.opacity = 1
            svg.style.color = 'red'
        else:
            button.style.opacity = ''
            svg.style.color = 'green'

        await asyncio.sleep(0.01)  # Yield to browser event loop


plugin = ReplStatus()
